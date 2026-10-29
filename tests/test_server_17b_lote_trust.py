"""GRNET fork: wallet attestations checked against a wallet providers LoTE."""

import base64
import json
from datetime import datetime, timedelta, timezone

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.x509.oid import NameOID
from cryptojwt.jwk.ec import ECKey
from cryptojwt.jws.jws import JWS

from idpyoidc.server.client_authn import _verify_jws_with_certificate
from idpyoidc.server.lote import WALLET_PROVIDER_ISSUANCE
from idpyoidc.server.lote import LoteError
from idpyoidc.server.lote import LoteTrustAnchors
from idpyoidc.server.lote import chain_is_trusted

NOW = datetime.now(timezone.utc)


def _name(cn):
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _cert(cn, key, issuer_cn=None, issuer_key=None, ca=False, days=(-1, 365)):
    issuer_key = issuer_key or key
    return (
        x509.CertificateBuilder()
        .subject_name(_name(cn))
        .issuer_name(_name(issuer_cn or cn))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(NOW + timedelta(days=days[0]))
        .not_valid_after(NOW + timedelta(days=days[1]))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
        .sign(issuer_key, hashes.SHA256())
    )


def _b64(cert):
    return base64.b64encode(cert.public_bytes(serialization.Encoding.DER)).decode()


def _b64url(data: bytes):
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


@pytest.fixture(scope="module")
def pki():
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca = _cert("Wallet Provider CA", ca_key, ca=True)
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf = _cert("Wallet Provider", leaf_key, "Wallet Provider CA", ca_key)
    other_key = ec.generate_private_key(ec.SECP256R1())
    other = _cert("Someone Else", other_key)
    return dict(ca=ca, ca_key=ca_key, leaf=leaf, leaf_key=leaf_key, other=other)


def test_leaf_under_listed_ca(pki):
    assert chain_is_trusted([pki["leaf"]], [pki["ca"]])
    assert chain_is_trusted([pki["leaf"], pki["ca"]], [pki["ca"]])


def test_listed_leaf(pki):
    assert chain_is_trusted([pki["leaf"]], [pki["leaf"]])


def test_unlisted(pki):
    assert not chain_is_trusted([pki["other"]], [pki["ca"]])
    assert not chain_is_trusted([pki["leaf"]], [])


def test_broken_chain(pki):
    # x5c claims other as the leaf's issuer: the signature doesn't match.
    assert not chain_is_trusted([pki["leaf"], pki["other"]], [pki["other"]])


def test_expired_leaf(pki):
    old = _cert("Old", pki["leaf_key"], "Wallet Provider CA", pki["ca_key"], days=(-30, -1))
    assert not chain_is_trusted([old], [pki["ca"]])


def test_attestation_signature(pki):
    key = ECKey(priv_key=pki["leaf_key"])
    token = JWS(json.dumps({"sub": "w"}), alg="ES256").sign_compact([key])
    _verify_jws_with_certificate(token, pki["leaf"])
    with pytest.raises(Exception):
        _verify_jws_with_certificate(token, pki["other"])


def test_attestation_signature_with_kid(pki):
    # Most wallets put a kid in the attestation header; the key built from the
    # certificate has none, and must still match.
    key = ECKey(priv_key=pki["leaf_key"], kid="DuVIGCpOg8UeaMf8SjJOedzjcZJ2AD0bJitilM3Yge4")
    token = JWS(json.dumps({"sub": "w"}), alg="ES256").sign_compact([key])
    _verify_jws_with_certificate(token, pki["leaf"])
    with pytest.raises(Exception):
        _verify_jws_with_certificate(token, pki["other"])


def _lote(entries, signer_key, signer_cert, next_update=None):
    payload = {
        "LoTE": {
            "ListAndSchemeInformation": {
                "NextUpdate": (next_update or NOW + timedelta(days=30)).isoformat()
            },
            "TrustedEntitiesList": entries,
        }
    }
    header = {"alg": "RS256", "x5c": [_b64(signer_cert)]}
    signing_input = f"{_b64url(json.dumps(header).encode())}.{_b64url(json.dumps(payload).encode())}"
    signature = signer_key.sign(signing_input.encode(), padding.PKCS1v15(), hashes.SHA256())
    return f"{signing_input}.{_b64url(signature)}"


def _entry(cert, service_type=WALLET_PROVIDER_ISSUANCE):
    return {
        "TrustedEntityServices": [
            {
                "ServiceInformation": {
                    "ServiceTypeIdentifier": service_type,
                    "ServiceDigitalIdentity": {"X509Certificates": [{"val": _b64(cert)}]},
                }
            }
        ]
    }


@pytest.fixture(scope="module")
def signer():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key, _cert("List Signer", key, ca=True)


def test_parse_takes_the_service_type(pki, signer, tmp_path):
    key, cert = signer
    ca_file = tmp_path / "ca.pem"
    ca_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    lote = _lote(
        [_entry(pki["ca"]), _entry(pki["other"], "http://uri.etsi.org/19602/SvcType/WalletSolution/Revocation")],
        key,
        cert,
    )
    anchors = LoteTrustAnchors("unused", signer_ca_path=str(ca_file)).parse(lote)
    assert [a.subject for a in anchors] == [pki["ca"].subject]


def test_parse_empty_twin(signer):
    key, cert = signer
    assert LoteTrustAnchors("unused").parse(_lote(None, key, cert)) == []


def test_parse_rejects_wrong_signer_ca(pki, signer, tmp_path):
    key, cert = signer
    ca_file = tmp_path / "other.pem"
    ca_file.write_bytes(pki["other"].public_bytes(serialization.Encoding.PEM))
    with pytest.raises(LoteError):
        LoteTrustAnchors("unused", signer_ca_path=str(ca_file)).parse(_lote([], key, cert))


def test_parse_rejects_tampered_list(pki, signer):
    key, cert = signer
    header, payload, signature = _lote([_entry(pki["ca"])], key, cert).split(".")
    forged = _b64url(json.dumps({"LoTE": {"TrustedEntitiesList": [_entry(pki["other"])]}}).encode())
    with pytest.raises(Exception):
        LoteTrustAnchors("unused").parse(f"{header}.{forged}.{signature}")


def test_parse_rejects_stale_list(signer):
    key, cert = signer
    with pytest.raises(LoteError):
        LoteTrustAnchors("unused").parse(_lote([], key, cert, NOW - timedelta(days=1)))
