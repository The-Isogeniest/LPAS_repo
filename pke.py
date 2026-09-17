"""
Public-key encryption scheme of Figure 11 (Appendix D): a simplified
Kyber-style Module-LWE public-key encryption scheme.

it has own modulus Q_PKE=7681 and module rank D=3
(Appendix E, "PKE parameters"), independent of the signature scheme's
modulus/rank in params.py. KeyGen/Encrypt/Decrypt below match Figure 11's
equations (b := A.s+e; c1 := A^T.y+e1, c2 := b^T.y+e2+Encode(m); v :=
c2-s^T.c1, m := Decode(v)) coefficient-for-coefficient.

"""
from __future__ import annotations
import hashlib
import numpy as np

from params import N
from ring import Poly

Q_PKE = 7681       # paper: Appendix E, "PKE parameters. d=3, q_PKE=7681"
D = 3              # module rank (paper: d)
TAU = 1            # l_infty bound on short PKE elements (T = {f : ||f||_inf <= 1})
DELTA = Q_PKE // 2  # Encode/Decode scaling factor, floor(q_PKE/2)


def _modq_pke(p: Poly) -> Poly:
    # reduce a Poly's coefficients mod Q_PKE, centered. Can't reuse
    # Poly.modq()/ring.matvec here: those always reduce mod the signature
    # scheme's Q, and the PKE lives at its own, separate modulus.
    r = np.mod(p.c, Q_PKE)
    r = np.where(r > Q_PKE // 2, r - Q_PKE, r)
    return Poly(r)


def _expand_matrix(seed: bytes) -> list:
    # ExpandA_PKE(seed): deterministically expand a seed into A in R_qPKE^{DxD}
    seed_int = int.from_bytes(hashlib.sha256(seed).digest()[:8], "big")
    rng = np.random.default_rng(seed_int)
    return [[Poly(rng.integers(0, Q_PKE, size=N)) for _ in range(D)] for _ in range(D)]


def _sample_short_vec(rng, length) -> list:
    # ternary coefficients, the T distribution for secrets/errors
    return [Poly(rng.integers(-TAU, TAU + 1, size=N)) for _ in range(length)]


def _matvec(mat, v) -> list:
    # A . v, reduced mod Q_PKE
    out = []
    for row in mat:
        acc = Poly.zero()
        for a_ij, v_j in zip(row, v):
            acc = acc + a_ij * v_j
        out.append(_modq_pke(acc))
    return out


def _matTvec(mat, v) -> list:
    # A^T . v, reduced mod Q_PKE
    out = []
    for col in range(len(mat[0])):
        acc = Poly.zero()
        for row in range(len(mat)):
            acc = acc + mat[row][col] * v[row]
        out.append(_modq_pke(acc))
    return out


def _dot(u, v) -> Poly:
    # inner product of two length-D vectors, reduced mod Q_PKE
    acc = Poly.zero()
    for a, b in zip(u, v):
        acc = acc + a * b
    return _modq_pke(acc)


def keygen(rng: np.random.Generator):
    # Figure 11 KeyGen: b := A.s + e
    seed = rng.bytes(32)
    A = _expand_matrix(seed)
    s = _sample_short_vec(rng, D)
    e = _sample_short_vec(rng, D)
    b = [_modq_pke(a_i + e_i) for a_i, e_i in zip(_matvec(A, s), e)]
    pk = (seed, b)
    sk = s
    return pk, sk


def encrypt(rng: np.random.Generator, pk, m: Poly):
    """Encrypts a single Poly of 0/1 coefficients (N message bits), matching
    Figure 11's Enc exactly."""
    seed, b = pk
    A = _expand_matrix(seed)
    y = _sample_short_vec(rng, D)
    e1 = _sample_short_vec(rng, D)
    e2 = Poly(rng.integers(-TAU, TAU + 1, size=N))
    c1 = [_modq_pke(c_i + e1_i) for c_i, e1_i in zip(_matTvec(A, y), e1)]
    encoded = Poly(DELTA * m.c)
    c2 = _modq_pke(_dot(b, y) + e2 + encoded)
    return (c1, c2)


def decrypt(sk, ct) -> Poly:
    """Inverse of encrypt(): recovers the single message Poly, matching
    Figure 11's Dec exactly."""
    s = sk
    c1, c2 = ct
    v = _modq_pke(c2 - _dot(s, c1))
    bits = np.round(v.c.astype(np.float64) / DELTA).astype(np.int64) % 2
    return Poly(bits)


# --------------------------------------------------------------------------
# Bit-string <-> message-polynomial chunking (not part of Figure 11 itself,
# just plumbing so a LAMBDA-bit string can be sent through encrypt/decrypt
# above even when N < LAMBDA, as in the toy profile).
# --------------------------------------------------------------------------

def bytes_to_plaintext_polys(data: bytes) -> list:
    # split a byte string into ceil(nbits/N) single Poly chunks, one bit per coefficient
    nbits = len(data) * 8
    n_polys = -(-nbits // N)  # ceil division
    total_slots = n_polys * N
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    if nbits < total_slots:
        bits = np.concatenate([bits, np.zeros(total_slots - nbits, dtype=np.uint8)])
    return [Poly(bits[i * N:(i + 1) * N].astype(np.int64)) for i in range(n_polys)]


def plaintext_polys_to_bytes(polys: list, nbytes: int) -> bytes:
    # inverse of bytes_to_plaintext_polys, reading back only the first nbytes
    bits = np.concatenate([p.c for p in polys]).astype(np.uint8)
    return np.packbits(bits[:nbytes * 8]).tobytes()


def encrypt_bytes(rng: np.random.Generator, pk, data: bytes) -> list:
    """Encrypts an arbitrary byte string as a list of independent Figure-11
    ciphertexts (one per N-bit chunk), each with its own fresh randomness,
    sharing one PKE keypair. When N >= len(data)*8 (e.g. the paper profile
    with N=LAMBDA=256), this is exactly one Figure-11 encrypt() call."""
    return [encrypt(rng, pk, m) for m in bytes_to_plaintext_polys(data)]


def decrypt_bytes(sk, ct_list: list, nbytes: int) -> bytes:
    # inverse of encrypt_bytes
    return plaintext_polys_to_bytes([decrypt(sk, ct) for ct in ct_list], nbytes)
