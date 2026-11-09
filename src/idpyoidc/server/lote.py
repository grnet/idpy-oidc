"""
GRNET fork: trust anchors from an ETSI TS 119 602 List of Trusted Entities (LoTE),
for checking wallet attestations against a wallet providers list such as the
European Commission's (https://development.trust.tech.ec.europa.eu/lists/eudiw/).

A LoTE is a compact JWS whose payload lists trusted entities and, for each, its
services with their certificates. The certificates of the services of the wanted
type are the trust anchors. They may be CAs or end-entity certificates: a chain
is trusted when one of its certificates is an anchor, or is directly issued by
one, and every certificate below it is directly issued by the next one up.
"""

import base64
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptojwt.jwk.ec import ECKey
from cryptojwt.jwk.rsa import RSAKey
from cryptojwt.jws.jws import factory

logger = logging.getLogger(__name__)

WALLET_PROVIDER_ISSUANCE = "http://uri.etsi.org/19602/SvcType/WalletSolution/Issuance"


class LoteError(Exception):
    pass


def _verify_jws(compact_jws: str) -> Tuple[List[x509.Certificate], dict]:
    """The signer's chain (x5c) and the payload, once the signature checks out."""
    jws = factory(compact_jws)
    if jws is None:
        raise LoteError("not a compact JWS")
    header = jws.jwt.headers
    x5c = [x509.load_der_x509_certificate(base64.b64decode(c)) for c in header.get("x5c", [])]
    if not x5c:
        raise LoteError("no x5c in the LoTE's JWS header")
    public_key = x5c[0].public_key()
    if isinstance(public_key, rsa.RSAPublicKey):
        key = RSAKey(pub_key=public_key)
    elif isinstance(public_key, ec.EllipticCurvePublicKey):
        key = ECKey(pub_key=public_key)
    else:
        raise LoteError(f"unsupported LoTE signing key: {type(public_key).__name__}")
    # The key comes from x5c. cryptojwt drops a key whose kid differs from the
    # header's, and the Commission's lists carry one.
    key.kid = header.get("kid") or ""
    payload = jws.verify_compact(compact_jws, keys=[key])
    # A dict, unless the header has a cty (the Commission's lists say "json").
    return x5c, json.loads(payload) if isinstance(payload, (str, bytes)) else payload


def _is_valid_now(cert: x509.Certificate, now: datetime) -> bool:
    return cert.not_valid_before_utc <= now <= cert.not_valid_after_utc


def _issued_by(cert: x509.Certificate, issuer: x509.Certificate) -> bool:
    try:
        cert.verify_directly_issued_by(issuer)
    except Exception:
        return False
    return True


def chain_is_trusted(chain: List[x509.Certificate], anchors: List[x509.Certificate],
                     now: Optional[datetime] = None) -> bool:
    """
    Whether chain (leaf first, as in x5c) leads to one of anchors. Checks
    signatures and validity periods; not revocation, which the wallet
    attestation's own status claim covers.
    """
    if not chain or not anchors:
        return False
    now = now or datetime.now(timezone.utc)
    anchor_der = {a.public_bytes(serialization.Encoding.DER) for a in anchors}
    for i, cert in enumerate(chain):
        if not _is_valid_now(cert, now):
            return False
        if i > 0 and not _issued_by(chain[i - 1], cert):
            return False
        # The anchor itself is in the chain.
        if cert.public_bytes(serialization.Encoding.DER) in anchor_der:
            return True
        # The anchor issued this certificate.
        if any(_issued_by(cert, anchor) and _is_valid_now(anchor, now) for anchor in anchors):
            return True
    return False


class LoteTrustAnchors:
    """
    The certificates of one service type in a LoTE, downloaded and verified on
    first use and again once ttl seconds have passed. When a refresh fails the
    previous anchors are kept, so a list host outage does not lock wallets out.
    """

    def __init__(self, url: str, service_type: str = WALLET_PROVIDER_ISSUANCE,
                 signer_ca_path: Optional[str] = None, ttl: int = 1200, timeout: int = 15):
        self.url = url
        self.service_type = service_type
        self.ttl = ttl
        self.timeout = timeout
        self.signer_cas = self._load_pem_bundle(signer_ca_path) if signer_ca_path else []
        self._anchors: List[x509.Certificate] = []
        self._loaded_at = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def _load_pem_bundle(path: str) -> List[x509.Certificate]:
        with open(path, "rb") as f:
            return x509.load_pem_x509_certificates(f.read())

    def anchors(self) -> List[x509.Certificate]:
        with self._lock:
            if time.time() - self._loaded_at > self.ttl:
                try:
                    self._anchors = self._fetch()
                    self._loaded_at = time.time()
                    logger.info(f"LoTE {self.url}: {len(self._anchors)} trust anchor(s)")
                except Exception as e:
                    if not self._anchors:
                        raise LoteError(f"LoTE {self.url} could not be loaded: {e}") from e
                    logger.warning(f"LoTE {self.url} refresh failed, keeping the previous anchors: {e}")
            return list(self._anchors)

    def _fetch(self) -> List[x509.Certificate]:
        response = requests.get(self.url, timeout=self.timeout)
        response.raise_for_status()
        return self.parse(response.text.strip())

    def parse(self, compact_jws: str) -> List[x509.Certificate]:
        x5c, payload = _verify_jws(compact_jws)
        # The signer must chain to a configured CA (the Commission's root and
        # CommisSign); without one configured only the signature is checked.
        if self.signer_cas and not chain_is_trusted(x5c, self.signer_cas):
            raise LoteError("the LoTE's signer does not chain to the configured CAs")

        lote = payload.get("LoTE", payload)
        next_update = lote.get("ListAndSchemeInformation", {}).get("NextUpdate")
        if next_update:
            when = datetime.fromisoformat(next_update.replace("Z", "+00:00"))
            if when < datetime.now(timezone.utc):
                raise LoteError(f"the LoTE is past its NextUpdate ({next_update})")

        anchors = []
        # The Commission publishes an empty copy of each list for negative tests
        # (wallet-providers_empty.json beside wallet-providers.json). The empty
        # copies leave TrustedEntitiesList out altogether.
        for entity in lote.get("TrustedEntitiesList") or []:
            for service in entity.get("TrustedEntityServices") or []:
                info = service.get("ServiceInformation", {})
                if info.get("ServiceTypeIdentifier") != self.service_type:
                    continue
                identity = info.get("ServiceDigitalIdentity", {})
                for cert in identity.get("X509Certificates") or []:
                    try:
                        anchors.append(x509.load_der_x509_certificate(base64.b64decode(cert["val"])))
                    except Exception as e:
                        logger.warning(f"LoTE {self.url}: skipping an unreadable certificate: {e}")
        return anchors


_instances: Dict[Tuple[str, str, Optional[str]], LoteTrustAnchors] = {}
_instances_lock = threading.Lock()


def lote_trust_anchors(url: str, service_type: str = WALLET_PROVIDER_ISSUANCE,
                       signer_ca_path: Optional[str] = None) -> LoteTrustAnchors:
    """One cached instance per list, shared by every request."""
    key = (url, service_type, signer_ca_path)
    with _instances_lock:
        if key not in _instances:
            _instances[key] = LoteTrustAnchors(url, service_type, signer_ca_path)
        return _instances[key]
