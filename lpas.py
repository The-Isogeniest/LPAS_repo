"""
LPAS: Lattice-based Proxy Adaptor Signatures, the algorithms of Figure 3.

Wires together the other modules:
  - ring.py       polynomial-ring / module arithmetic
  - signature.py  the Fiat-Shamir-with-aborts scheme Pi (Figure 2) and Rej
  - pke.py        the MLWE-based PKE (Figure 11)
  - ske.py        AES-256-GCM
  - nizk.py       a non-zero-knowledge stub standing in for the real NIZK

Naming follows the paper: ReqGen/ReqVerify/AdGen/AdVerify/ProxySign/
ProxyPreVerify/Adapt/ProxyExt/ReqExt, plus Setup. Unlike the paper, public
parameters `pp` are passed explicitly instead of being implicit global state.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import params as P
import pke as PKE
import ske as SKE
import nizk as NIZK
import signature as SIG
from ring import (
    Poly, vec_add, vec_sub, vec_linf, vec_l2, matvec,
    scalar_vec_mul, scale_vec, vec_pack_bits, vec_unpack_bits, sample_ternary_vec,
    sample_gauss_vec, round_nu, center_mod_vec, combine_rounded, bits_needed,
)
from signature import H_bytes, H_challenge, vk_to_bytes, xor_bytes, rej_accept


def _q_nu(nu: int) -> int:
    return P.Q // (1 << nu) if nu > 0 else P.Q


# Fixed, publicly-known bit-width for the blindness r (and the blinded
# witness wit' = wit + r): both are bounded by the public norm bound
# B_wit, so this is a real wire-format width, not one measured from any
# particular run -- see the r/wit' packing in ad_gen/req_ext below, which
# is the one spot in this protocol where such a vector actually gets
# serialized to bytes (for the SKE ciphertext), rather than passed around
# as a live Python object like z/h/wit' elsewhere.
WIT_BITS = bits_needed(int(np.ceil(P.B_WIT)))


# --------------------------------------------------------------------------
# The hard relation R_A: A = [Abar | I_k], R_A = {(t, r) : A.r = t (mod q)
# and ||r||_inf <= 1}. Same matrix as the proxy's verification key.
# --------------------------------------------------------------------------

def in_relation_A(A, stmt, wit) -> bool:
    # membership test for R_A: witness is short, and A.wit lands on stmt
    if vec_linf(wit) > 1:
        return False
    return all(p == q for p, q in zip(matvec(A, wit), stmt))


def gen_relation_A(rng, A):
    # demo helper: sample a fresh (stmt, wit) pair in R_A for the seller
    wit = sample_ternary_vec(rng, P.M)
    stmt = matvec(A, wit)
    return stmt, wit


# --------------------------------------------------------------------------
# Public parameters
# --------------------------------------------------------------------------

@dataclass
class PublicParams:
    crs: object
    vk: tuple      # Pi's verification key (seed, b)
    A: list        # the shared public matrix (paper: Abar / A)


def setup(rng: np.random.Generator):
    # generate the proxy's signing key and bundle the public parts into pp
    crs = NIZK.setup()
    vk, sk, A = SIG.keygen(rng)
    pp = PublicParams(crs=crs, vk=vk, A=A)
    return pp, sk


# --------------------------------------------------------------------------
# Buyer: ReqGen / ReqVerify, and later ReqExt
# --------------------------------------------------------------------------

def req_gen(rng, stmt):
    # buyer mints a fresh PKE keypair and requests the statement it wants
    pke_pk, pke_sk = PKE.keygen(rng)
    req = (pke_pk, stmt)
    st_buyer = pke_sk
    return req, st_buyer


def req_verify(stmt, req) -> bool:
    # check the request actually names the statement it claims to
    _, stmt_in_req = req
    return stmt == stmt_in_req


# --------------------------------------------------------------------------
# Seller: AdGen, and later Adapt
# --------------------------------------------------------------------------

def ad_gen(rng, pp: PublicParams, req, wit):
    # seller blinds its witness, encrypts a one-time key and the blindness, and builds the advertisement
    pke_pk, stmt = req
    for _ in range(P.MAX_REJ_TRIES):
        r = sample_gauss_vec(rng, P.M, P.SIGMA_WIT)
        if not rej_accept(rng, wit, r, P.SIGMA_WIT, P.REJ_M_WIT):
            continue
        wit_blinded = vec_add(wit, r)

        k = rng.bytes(P.LAMBDA // 8)
        u = xor_bytes(k, H_bytes(*[p.tobytes() for p in wit_blinded]))
        c1 = PKE.encrypt_bytes(rng, pke_pk, u)
        c2 = SKE.encrypt(k, vec_pack_bits(r, WIT_BITS))
        ct = (c1, c2)
        t = matvec(pp.A, r)

        statement = NIZK.Statement(A=pp.A, stmt=stmt, t=t, pke_pk=pke_pk, ct=ct)
        witness = NIZK.Witness(wit=wit, wit_blinded=wit_blinded, r=r, k=k, u=u)
        pi_advt = NIZK.prove(pp.crs, statement, witness)

        advt = (pi_advt, t, ct)
        st_advt = r
        return advt, st_advt
    raise RuntimeError("AdGen: rejection sampling did not accept within MAX_REJ_TRIES")


def ad_verify(pp: PublicParams, req, advt) -> bool:
    # proxy checks the advertisement's proof without seeing the witness
    pke_pk, stmt = req
    pi_advt, t, ct = advt
    statement = NIZK.Statement(A=pp.A, stmt=stmt, t=t, pke_pk=pke_pk, ct=ct)
    return NIZK.verify(pp.crs, statement, pi_advt)


# --------------------------------------------------------------------------
# Proxy: ProxySign, ProxyPreVerify, ProxyExt
# --------------------------------------------------------------------------

def _reconstruct_w(pp: PublicParams, c: Poly, z, stmt, t):
    # shared helper for ProxySign / ProxyPreVerify: rebuild the commitment w
    _, b = pp.vk
    cb_scaled = scale_vec(scalar_vec_mul(c, b), 1 << P.NU_B)
    return round_nu(vec_add(vec_sub(matvec(pp.A, z), cb_scaled), vec_add(stmt, t)), P.NU_W)


def proxy_sign(rng, pp: PublicParams, sk, msg: bytes, req, advt):
    # like Sign, but the commitment is shifted by stmt+t to bind the pre-signature to this purchase
    pke_pk, stmt = req
    pi_advt, t, ct = advt
    assert ad_verify(pp, req, advt), "ProxySign: invalid advertisement"

    for _ in range(P.MAX_REJ_TRIES):
        y = sample_gauss_vec(rng, P.M, P.SIGMA_Y_TILDE)
        w = round_nu(vec_add(matvec(pp.A, y), vec_add(stmt, t)), P.NU_W)
        c = H_challenge(vk_to_bytes(pp.vk), msg, w)
        cs = scalar_vec_mul(c, sk)
        if not rej_accept(rng, cs, y, P.SIGMA_Y_TILDE, P.REJ_M_Y_TILDE):
            continue
        z_tilde = vec_add(cs, y)
        h_raw = vec_sub(w, _reconstruct_w(pp, c, z_tilde, stmt, t))
        h = center_mod_vec(h_raw, _q_nu(P.NU_W))
        return (c, z_tilde, h)
    raise RuntimeError("ProxySign: rejection sampling did not accept within MAX_REJ_TRIES")


def proxy_pre_verify(pp: PublicParams, msg: bytes, req, advt, presig) -> bool:
    # rebuild the commitment from the pre-signature and check challenge + norm bound
    _, stmt = req
    _, t, _ = advt
    c, z_tilde, h = presig
    w = combine_rounded(_reconstruct_w(pp, c, z_tilde, stmt, t), h, P.NU_W)
    if H_challenge(vk_to_bytes(pp.vk), msg, w) != c:
        return False
    # norm bound is the SUM ||z~||_2 + ||2^nu_w . h||_2 <= B_proxysign (paper's ProxyPreVerify, step 5)
    h_scaled = scale_vec(h, 1 << P.NU_W)
    norm = vec_l2(z_tilde) + vec_l2(h_scaled)
    return norm <= P.B_PROXYSIGN


def proxy_ext(presig_full, presig):
    # subtracting the two z's cancels everything but the blinded witness wit'
    c1, z, _h1 = presig_full
    c2, z_tilde, _h2 = presig
    assert c1 == c2, "ProxyExt: challenges of sigma_sig and sigma~_sig must match"
    return vec_sub(z, z_tilde)


# --------------------------------------------------------------------------
# Seller: Adapt
# --------------------------------------------------------------------------

def adapt(pp: PublicParams, msg: bytes, req, advt, presig, wit, st_advt):
    # fold the seller's witness into the pre-signature, turning it into a full, verifiable signature
    assert proxy_pre_verify(pp, msg, req, advt, presig), "Adapt: invalid proxy pre-signature"
    c, z_tilde, h = presig
    r = st_advt
    wit_blinded = vec_add(wit, r)
    z = vec_add(z_tilde, wit_blinded)
    return (c, z, h)


# --------------------------------------------------------------------------
# Buyer: ReqExt
# --------------------------------------------------------------------------

def req_ext(pp: PublicParams, wit_blinded, st_buyer, stmt, advt):
    # buyer decrypts the one-time key using wit', recovers r, and checks the resulting witness
    pke_sk = st_buyer
    _, _, ct = advt
    c1, c2 = ct
    u = PKE.decrypt_bytes(pke_sk, c1, P.LAMBDA // 8)
    k = xor_bytes(u, H_bytes(*[p.tobytes() for p in wit_blinded]))
    try:
        r = vec_unpack_bits(SKE.decrypt(k, c2), P.M, WIT_BITS)
    except Exception:
        return None
    wit_star = vec_sub(wit_blinded, r)
    if in_relation_A(pp.A, stmt, wit_star):
        return wit_star
    return None
