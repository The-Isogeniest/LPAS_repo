"""
STUB for the NIZK argument system used by AdGen / AdVerify (Figure 3). This
is intentionally NOT a real zero-knowledge proof: the "prover" just hands
over the witness directly, and the "verifier" recomputes and checks every
equation in the clear.

Soundness is preserved (it checks more than a real proof would, since it
has the actual witness). Zero-knowledge is NOT preserved — this leaks the
witness to whoever calls AdVerify. Replace Prove/Verify with a real NIZK
before using this for anything beyond exercising the control flow.
"""
from __future__ import annotations
from dataclasses import dataclass

from ring import matvec, vec_add
from signature import H_bytes, xor_bytes


@dataclass
class Witness:
    # the private inputs the proof is about
    wit: list
    wit_blinded: list  # wit'
    r: list
    k: bytes
    u: bytes


@dataclass
class Statement:
    # the public inputs the proof is about
    A: list        # public matrix (paper: A = [Abar | I_k])
    stmt: list
    t: list
    pke_pk: object
    ct: tuple      # (c1, c2)


def setup():
    # no real CRS needed for this stub
    return None


def prove(crs, statement: Statement, witness: Witness):
    # NOT zero-knowledge: just hands back the witness as the "proof"
    return witness


def verify(crs, statement: Statement, proof: Witness) -> bool:
    # check the algebraic relations directly against the (fully revealed) witness
    w = proof
    A, stmt, t = statement.A, statement.stmt, statement.t

    # A . wit = stmt
    if any(p != q for p, q in zip(matvec(A, w.wit), stmt)):
        return False
    # A . r = t
    if any(p != q for p, q in zip(matvec(A, w.r), t)):
        return False
    # wit' = wit + r
    if any(p != q for p, q in zip(w.wit_blinded, vec_add(w.wit, w.r))):
        return False
    # u = k xor H(wit')
    if w.u != xor_bytes(w.k, H_bytes(*[p.tobytes() for p in w.wit_blinded])):
        return False
    return True
