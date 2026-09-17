"""
Symmetric-key encryption SKE, instantiated with AES-256-GCM as the paper
specifies. The key is always the LAMBDA=256-bit string produced by the
LPAS protocol, matching AES-256's key size.

Uses a random 96-bit nonce per encryption, prepended to the ciphertext.
Decryption with the wrong key raises rather than returning garbage.
"""
from __future__ import annotations
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

NONCE_LEN = 12  # 96 bits, the standard AES-GCM nonce size


def encrypt(key: bytes, plaintext: bytes, associated_data: bytes = b"") -> bytes:
    # encrypt with a fresh random nonce, prepended to the output
    assert len(key) == 32, "SKE is instantiated with AES-256, expected a 32-byte key"
    nonce = os.urandom(NONCE_LEN)
    ct = AESGCM(key).encrypt(nonce, plaintext, associated_data)
    return nonce + ct


def decrypt(key: bytes, ciphertext: bytes, associated_data: bytes = b"") -> bytes:
    # split the nonce back off and decrypt; wrong key raises instead of returning garbage
    assert len(key) == 32, "SKE is instantiated with AES-256, expected a 32-byte key"
    nonce, ct = ciphertext[:NONCE_LEN], ciphertext[NONCE_LEN:]
    return AESGCM(key).decrypt(nonce, ct, associated_data)
