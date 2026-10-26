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
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

logger = logging.getLogger(__name__)

WALLET_PROVIDER_ISSUANCE = "http://uri.etsi.org/19602/SvcType/WalletSolution/Issuance"

_JWS_HASHES = {"256": hashes.SHA256(), "384": hashes.SHA384(), "512": hashes.SHA512()}


class LoteError(Exception):
    pass


def _b64url(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _verify_jws_signature(header: dict, signing_input: bytes, signature: bytes,
                          cert: x509.Certificate) -> None:
    alg = header.get("alg", "")
    digest = _JWS_HASHES.get(alg[2:])
    key = cert.public_key()
    if alg.startswith("RS") and digest and isinstance(key, rsa.RSAPublicKey):
        key.verify(signature, signing_input, padding.PKCS1v15(), digest)
    elif alg.startswith("PS") and digest and isinstance(key, rsa.RSAPublicKey):
        key.verify(signature, signing_input,
                   padding.PSS(mgf=padding.MGF1(digest), salt_length=digest.digest_size),
                   digest)
    elif alg.startswith("ES") and digest and isinstance(key, ec.EllipticCurvePublicKey):
        half = len(signature) // 2
        der = encode_dss_signature(int.from_bytes(signature[:half], "big"),
                                   int.from_bytes(signature[half:], "big"))
        key.verify(der, signing_input, ec.ECDSA(digest))
    else:
        raise LoteError(f"unsupported LoTE signature: alg {alg}, key {type(key).__name__}")


def _is_valid_now(cert: x509.Certificate, now: datetime) -> bool:
    return cert.not_valid_before_utc <= now <= cert.not_valid_after_utc


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
        if i > 0:
            try:
                chain[i - 1].verify_directly_issued_by(cert)
            except Exception:
                return False
        # The anchor itself is in the chain.
        if cert.public_bytes(serialization.Encoding.DER) in anchor_der:
            return True
        # The anchor issued this certificate.
        for anchor in anchors:
            try:
                cert.verify_directly_issued_by(anchor)
            except Exception:
                continue
            if _is_valid_now(anchor, now):
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
        parts = compact_jws.split(".")
        if len(parts) != 3:
            raise LoteError("not a compact JWS")
        header = json.loads(_b64url(parts[0]))
        x5c = [x509.load_der_x509_certificate(base64.b64decode(c)) for c in header.get("x5c", [])]
        if not x5c:
            raise LoteError("no x5c in the LoTE's JWS header")
        _verify_jws_signature(header, f"{parts[0]}.{parts[1]}".encode("ascii"),
                              _b64url(parts[2]), x5c[0])
        # The signer must chain to a configured CA (the Commission's root and
        # CommisSign); without one configured only the signature is checked.
        if self.signer_cas and not chain_is_trusted(x5c, self.signer_cas):
            raise LoteError("the LoTE's signer does not chain to the configured CAs")

        payload = json.loads(_b64url(parts[1]))
        lote = payload.get("LoTE", payload)
        next_update = lote.get("ListAndSchemeInformation", {}).get("NextUpdate")
        if next_update:
            when = datetime.fromisoformat(next_update.replace("Z", "+00:00"))
            if when < datetime.now(timezone.utc):
                raise LoteError(f"the LoTE is past its NextUpdate ({next_update})")

        anchors = []
        # The empty twins used for negative tests have no TrustedEntitiesList.
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
