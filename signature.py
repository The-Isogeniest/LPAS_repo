"""
The underlying "Fiat-Shamir with aborts" signature scheme Pi = (KeyGen, Sign,
Verify) of Figure 2, together with the rejection-sampling algorithm Rej of
Figure 4.

Includes an optional modulus-rounding / key-compression step controlled by
(nu_b, nu_w), the same idea CRYSTALS-Dilithium uses to shrink keys and
signatures. The toy profile disables it (nu_b=nu_w=0), so the reconstruction
hint h is always zero; the paper profile turns it on.
"""
from __future__ import annotations
import hashlib
import numpy as np

import params as P
from ring import (
    Poly, vec_add, vec_sub, vec_l2, matvec,
    scalar_vec_mul, scale_vec, vec_to_bytes, sample_ternary_vec, sample_gauss_vec,
    expand_matrix, round_nu, center_mod_vec, combine_rounded,
)


def _q_nu(nu: int) -> int:
    # the smaller modulus a rounded value lives in once nu bits are dropped
    return P.Q // (1 << nu) if nu > 0 else P.Q


# --------------------------------------------------------------------------
# Hashing / challenge generation
# --------------------------------------------------------------------------

def _sample_in_ball(digest: bytes) -> Poly:
    # turn a hash digest into a sparse +-1 challenge polynomial with exactly OMEGA nonzero coefficients
    seed_int = int.from_bytes(digest, "big")
    rng = np.random.default_rng(seed_int % (2 ** 63))
    coeffs = np.zeros(P.N, dtype=np.int64)
    positions = rng.choice(P.N, size=P.OMEGA, replace=False)
    signs = rng.choice([-1, 1], size=P.OMEGA)
    coeffs[positions] = signs
    return Poly(coeffs)


def H_challenge(vk_bytes: bytes, msg: bytes, w) -> Poly:
    # the Fiat-Shamir challenge: hash (vk, msg, commitment) into a challenge polynomial
    digest = hashlib.sha256(vk_bytes + msg + vec_to_bytes(w)).digest()
    return _sample_in_ball(digest)


def H_bytes(*chunks: bytes) -> bytes:
    # generic hash to LAMBDA bits, built on SHA-256
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    digest = h.digest()
    assert P.LAMBDA % 8 == 0
    out = digest
    while len(out) < P.LAMBDA // 8:
        h = hashlib.sha256(out)
        out += h.digest()
    return out[: P.LAMBDA // 8]


def xor_bytes(a: bytes, b: bytes) -> bytes:
    # byte-for-byte XOR
    return bytes(x ^ y for x, y in zip(a, b))


# --------------------------------------------------------------------------
# Rejection sampling, Figure 4: Rej(v, chi_r, chi_z, M; r) decides whether to
# accept z = v + r. Rejecting means the caller must redraw r and retry the
# whole step (see sign()/proxy_sign()/ad_gen()'s retry loops).
# --------------------------------------------------------------------------

def rej_accept(rng, v, r, sigma: float, M_rej: float) -> bool:
    # one trial of Figure 4's Rej: decide whether to accept z = v + r
    v_flat = np.concatenate([p.c for p in v]).astype(np.float64)
    r_flat = np.concatenate([p.c for p in r]).astype(np.float64)
    exponent = -(np.dot(v_flat, v_flat) + 2.0 * np.dot(v_flat, r_flat)) / (2.0 * sigma * sigma)
    accept_prob = 1.0 if exponent > 0 else min(1.0, float(np.exp(exponent)) / M_rej)
    return bool(rng.random() < accept_prob)


# --------------------------------------------------------------------------
# Pi = (KeyGen, Sign, Verify), Figure 2
# --------------------------------------------------------------------------

def vk_to_bytes(vk) -> bytes:
    # serialize a verification key for hashing
    seed, b = vk
    return seed + vec_to_bytes(b)


def keygen(rng: np.random.Generator):
    # build a fresh (verification key, signing key) pair
    seed = rng.bytes(16)
    Abar_pub = expand_matrix(seed, P.K, P.L)                    # A in R_q^{k x l}
    # build Abar = [A | I_k] in R_q^{k x (l+k)}
    identity_rows = []
    for i in range(P.K):
        row = [Poly.zero() for _ in range(P.K)]
        one = np.zeros(P.N, dtype=np.int64)
        one[0] = 1
        row[i] = Poly(one)
        identity_rows.append(row)
    Abar = [Abar_pub[i] + identity_rows[i] for i in range(P.K)]

    s = sample_ternary_vec(rng, P.M)                            # secret key
    b_hat = matvec(Abar, s)
    b = round_nu(b_hat, P.NU_B)                                 # nu_b = 0 => plain mod-q reduction
    vk = (seed, b)
    sk = s
    return vk, sk, Abar


def sign(rng, vk, sk, Abar, msg: bytes):
    # produce a signature on msg, retrying whenever rejection sampling rejects
    _, b = vk
    for _ in range(P.MAX_REJ_TRIES):
        y = sample_gauss_vec(rng, P.M, P.SIGMA_Y)
        w = round_nu(matvec(Abar, y), P.NU_W)
        c = H_challenge(vk_to_bytes(vk), msg, w)
        cs = scalar_vec_mul(c, sk)
        if not rej_accept(rng, cs, y, P.SIGMA_Y, P.REJ_M_Y):
            continue
        z = vec_add(cs, y)
        # h = w - round(A.z - 2^nu_b * c*b); zero when compression is disabled
        cb_scaled = scale_vec(scalar_vec_mul(c, b), 1 << P.NU_B)
        h_raw = vec_sub(w, round_nu(vec_sub(matvec(Abar, z), cb_scaled), P.NU_W))
        h = center_mod_vec(h_raw, _q_nu(P.NU_W))
        return (c, z, h)
    raise RuntimeError("Sign: rejection sampling did not accept within MAX_REJ_TRIES")


def verify(vk, Abar, msg: bytes, sig) -> bool:
    # check a signature: rebuild the commitment, recompute the challenge, check the norm bound
    _, b = vk
    c, z, h = sig
    cb_scaled = scale_vec(scalar_vec_mul(c, b), 1 << P.NU_B)
    u = round_nu(vec_sub(matvec(Abar, z), cb_scaled), P.NU_W)
    w = combine_rounded(u, h, P.NU_W)
    c_check = H_challenge(vk_to_bytes(vk), msg, w)
    if c_check != c:
        return False
    # h lives at the compressed scale, so scale it back up before the norm check.
    # Norm bound is the SUM ||z||_2 + ||2^nu_w . h||_2 <= B_sign (paper's Verify, step 7),
    # not a combined sqrt(||z||^2 + ||h||^2).
    h_scaled = scale_vec(h, 1 << P.NU_W)
    norm = vec_l2(z) + vec_l2(h_scaled)
    return norm <= P.B_SIGN
