"""
Arithmetic for the ring R = Z[x]/(x^n+1) and the module R^m, plus sampling
and modulus rounding.

Note: a Poly always keeps exact integer coefficients, never auto-reduced
mod q. "Public" values (matrices, keys, commitments) get reduced mod q
explicitly via .modq(); "short" values (secrets, randomness, signature
responses) are kept exact since their integer size is what matters for
security.
"""
from __future__ import annotations
import hashlib
from math import comb
import numpy as np

from params import N, Q


def _centered_mod(x: np.ndarray, q: int) -> np.ndarray:
    # fold into (-q/2, q/2] instead of [0, q)
    r = np.mod(x, q)
    return np.where(r > q // 2, r - q, r)


class Poly:
    """A single element of Z[x]/(x^n+1), stored as exact integer coefficients."""

    __slots__ = ("c",)

    def __init__(self, coeffs):
        c = np.asarray(coeffs, dtype=np.int64)
        assert c.shape == (N,), f"expected {N} coefficients, got {c.shape}"
        self.c = c

    @staticmethod
    def zero() -> "Poly":
        # the zero polynomial
        return Poly(np.zeros(N, dtype=np.int64))

    def modq(self) -> "Poly":
        # reduce down into R_q
        return Poly(_centered_mod(self.c, Q))

    def __add__(self, other: "Poly") -> "Poly":
        return Poly(self.c + other.c)

    def __sub__(self, other: "Poly") -> "Poly":
        return Poly(self.c - other.c)

    def __neg__(self) -> "Poly":
        return Poly(-self.c)

    def __mul__(self, other: "Poly") -> "Poly":
        # multiply two ring elements: convolve, then wrap the top half back with a minus sign (x^n = -1)
        conv = np.convolve(self.c, other.c)
        out = conv[:N].copy()
        out[:N - 1] -= conv[N:]
        return Poly(out)

    def __eq__(self, other: object) -> bool:
        # exact coefficient match
        return isinstance(other, Poly) and np.array_equal(self.c, other.c)

    def linf(self) -> int:
        # largest coefficient by absolute value
        return int(np.max(np.abs(self.c))) if N else 0

    def l2sq(self) -> int:
        # squared length of this one polynomial
        return int(np.sum(self.c.astype(np.int64) ** 2))

    def tobytes(self) -> bytes:
        # dump coefficients as fixed-width bytes
        return self.c.astype("<i8").tobytes()


# --------------------------------------------------------------------------
# Vectors of Poly (elements of R^m) and matrices (list of rows of Poly)
# --------------------------------------------------------------------------

def vec_add(u, v):
    # add two vectors of polynomials elementwise
    return [a + b for a, b in zip(u, v)]


def vec_sub(u, v):
    # subtract two vectors of polynomials elementwise
    return [a - b for a, b in zip(u, v)]


def scalar_vec_mul(c: Poly, v):
    # multiply every entry of a vector by the same polynomial
    return [c * vi for vi in v]


def scale_vec(v, factor: int):
    # multiply every coefficient of every entry by a plain integer (not a ring multiply)
    return [Poly(factor * p.c) for p in v]


def vec_modq(v):
    # reduce every entry of a vector mod q
    return [p.modq() for p in v]


def center_mod_vec(v, modulus: int):
    # reduce every entry into a centered range mod `modulus`
    return [Poly(_centered_mod(p.c, modulus)) for p in v]


def unsigned_mod_vec(v, modulus: int):
    # reduce every entry into the plain [0, modulus) range
    return [Poly(np.mod(p.c, modulus)) for p in v]


def combine_rounded(u, h, nu: int):
    # add a rounded value and its hint back together, in the same representation round_nu would produce
    summed = vec_add(u, h)
    if nu == 0:
        return vec_modq(summed)
    q_nu = Q // (1 << nu)
    return unsigned_mod_vec(summed, q_nu)


def vec_linf(v) -> int:
    # largest single coefficient across the whole vector
    return max((p.linf() for p in v), default=0)


def vec_l2(v) -> float:
    # Euclidean norm of the whole vector, pooled across all its polynomials
    return float(np.sqrt(sum(p.l2sq() for p in v)))


def matvec(mat, v):
    # public matrix-vector product over R_q
    out = []
    for row in mat:
        acc = Poly.zero()
        for a_ij, v_j in zip(row, v):
            acc = acc + a_ij * v_j
        out.append(acc.modq())
    return out


def matmat(mat_a, mat_b):
    # matrix-matrix product over R_q
    rows, cols = len(mat_a), len(mat_b[0])
    out = []
    for i in range(rows):
        row_out = []
        for j in range(cols):
            acc = Poly.zero()
            for k in range(len(mat_b)):
                acc = acc + mat_a[i][k] * mat_b[k][j]
            row_out.append(acc.modq())
        out.append(row_out)
    return out


def vec_to_bytes(v) -> bytes:
    # serialize a vector of polynomials to bytes
    return b"".join(p.tobytes() for p in v)


def bytes_to_vec(data: bytes, length: int):
    # inverse of vec_to_bytes
    arr = np.frombuffer(data, dtype="<i8")
    assert arr.size == length * N
    return [Poly(arr[i * N:(i + 1) * N].copy()) for i in range(length)]


def vec_pack_bits(v, bit_width: int) -> bytes:
    # like vec_to_bytes, but tight-packed at bit_width bits/coefficient
    # instead of a fixed 8 bytes/coefficient -- for short vectors (bounded
    # by a known norm bound) that need an actually compact wire encoding
    return pack_bits([int(x) for p in v for x in p.c], bit_width)


def vec_unpack_bits(data: bytes, length: int, bit_width: int):
    # inverse of vec_pack_bits, for a vector of `length` Poly
    flat = unpack_bits(data, length * N, bit_width)
    return [Poly(np.array(flat[i * N:(i + 1) * N], dtype=np.int64)) for i in range(length)]


# --------------------------------------------------------------------------
# Compact packing helpers for hex dumps / serialization (not part of the
# paper's algorithms, just demo/display tooling).
# --------------------------------------------------------------------------

def pack_ternary(values) -> bytes:
    # pack {-1,0,1} values 5-per-byte (base-3 digits)
    trits = [int(v) % 3 for v in values]  # -1 -> 2, 0 -> 0, 1 -> 1
    out = bytearray()
    for i in range(0, len(trits), 5):
        chunk = trits[i:i + 5]
        out.append(sum(t * (3 ** j) for j, t in enumerate(chunk)))
    return bytes(out)


def unpack_ternary(data: bytes, count: int):
    # inverse of pack_ternary
    trits = []
    for byte in data:
        b = byte
        for _ in range(5):
            trits.append(b % 3)
            b //= 3
    return [t - 3 if t == 2 else t for t in trits[:count]]


# --------------------------------------------------------------------------
# Sparse encoding for the challenge polynomial c (Table 1's challenge space
# C): exactly omega of its n coefficients are nonzero, each +-1, everything
# else exactly 0. Rather than a trit per coefficient (pack_ternary, which
# doesn't know about that sparsity), rank the SUPPORT -- which omega of the
# n positions are nonzero -- among the C(n,omega) possibilities via the
# standard combinatorial number system, and append omega sign bits. Matches
# the paper's own sizing (Appendix E): |c| = ceil(log2 C(n,omega)) + omega
# bits, the information-theoretic minimum for an element of C.
# --------------------------------------------------------------------------

def _comb_rank(positions_desc, k: int) -> int:
    # positions_desc: k distinct ints, sorted descending. Combinatorial
    # number system: rank = sum(C(a_i, k-i)) for the i-th largest position a_i
    return sum(comb(a, k - i) for i, a in enumerate(positions_desc))


def _comb_unrank(rank: int, n: int, k: int) -> list:
    # inverse of _comb_rank; returns k positions in [0,n), sorted descending
    positions = []
    remaining = rank
    a = n - 1
    for i in range(k):
        while comb(a, k - i) > remaining:
            a -= 1
        positions.append(a)
        remaining -= comb(a, k - i)
        a -= 1
    return positions


def pack_challenge(coeffs, omega: int) -> bytes:
    """Encodes a length-n challenge polynomial (coeffs: exactly `omega`
    entries in {-1,1}, the rest 0) as (support rank, sign bits) rather than
    one trit per coefficient. Raises if coeffs doesn't have exactly omega
    nonzero, +-1 entries."""
    n = len(coeffs)
    positions = [i for i, x in enumerate(coeffs) if x != 0]
    assert len(positions) == omega, f"expected exactly {omega} nonzero coefficients, got {len(positions)}"
    assert all(coeffs[i] in (-1, 1) for i in positions), "nonzero challenge coefficients must be +-1"
    rank = _comb_rank(sorted(positions, reverse=True), omega)
    signs = 0
    for i in sorted(positions):
        signs = (signs << 1) | (1 if coeffs[i] == -1 else 0)
    rank_bits = max(1, (comb(n, omega) - 1).bit_length())
    combined = (rank << omega) | signs
    nbytes = (rank_bits + omega + 7) // 8
    return combined.to_bytes(nbytes, "big")


def unpack_challenge(data: bytes, n: int, omega: int) -> list:
    """Inverse of pack_challenge: returns the n challenge coefficients."""
    rank_bits = max(1, (comb(n, omega) - 1).bit_length())
    combined = int.from_bytes(data, "big")
    signs = combined & ((1 << omega) - 1)
    rank = combined >> omega
    positions_asc = sorted(_comb_unrank(rank, n, omega))
    coeffs = [0] * n
    for idx, i in enumerate(positions_asc):
        bit = (signs >> (omega - 1 - idx)) & 1
        coeffs[i] = -1 if bit else 1
    return coeffs


def bits_needed(abs_bound: int) -> int:
    # smallest signed bit-width that fits +-abs_bound
    width = 1
    while (1 << (width - 1)) - 1 < abs_bound:
        width += 1
    return width


def pack_bits(values, bit_width: int) -> bytes:
    # pack signed integers into a bitstream, bit_width bits each
    lo, hi = -(1 << (bit_width - 1)), (1 << (bit_width - 1)) - 1
    bits = []
    for v in values:
        v = int(v)
        assert lo <= v <= hi, f"value {v} does not fit in {bit_width} signed bits"
        u = v & ((1 << bit_width) - 1)
        bits.extend((u >> (bit_width - 1 - i)) & 1 for i in range(bit_width))
    while len(bits) % 8:
        bits.append(0)
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i + 8]:
            byte = (byte << 1) | b
        out.append(byte)
    return bytes(out)


def unpack_bits(data: bytes, count: int, bit_width: int):
    # inverse of pack_bits
    bits = []
    for byte in data:
        bits.extend((byte >> (7 - i)) & 1 for i in range(8))
    out = []
    for i in range(count):
        u = 0
        for b in bits[i * bit_width:(i + 1) * bit_width]:
            u = (u << 1) | b
        if u >= (1 << (bit_width - 1)):
            u -= (1 << bit_width)
        out.append(u)
    return out


def base_q_group_bytes(q: int, group_size: int) -> int:
    # bytes needed to store `group_size` coefficients mod q packed together
    bits = (q ** group_size - 1).bit_length()
    return (bits + 7) // 8


def best_group_size(q: int, max_group: int = 8) -> int:
    # pick the group size that wastes the fewest bits per coefficient
    return min(range(1, max_group + 1), key=lambda k: base_q_group_bytes(q, k) / k)


def pack_modq(values, q: int, group_size: int) -> bytes:
    # pack mod-q coefficients, group_size at a time, as base-q big integers
    nbytes = base_q_group_bytes(q, group_size)
    out = bytearray()
    for i in range(0, len(values), group_size):
        chunk = [int(v) % q for v in values[i:i + group_size]]
        combined = 0
        for c in reversed(chunk):
            combined = combined * q + c
        out += combined.to_bytes(nbytes, "big")
    return bytes(out)


def unpack_modq(data: bytes, count: int, q: int, group_size: int):
    # inverse of pack_modq
    nbytes = base_q_group_bytes(q, group_size)
    out = []
    for i in range(0, len(data), nbytes):
        combined = int.from_bytes(data[i:i + nbytes], "big")
        for _ in range(group_size):
            out.append(combined % q)
            combined //= q
    return out[:count]


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def sample_uniform_poly(rng: np.random.Generator) -> Poly:
    # coefficients uniform over all of Z_q
    return Poly(rng.integers(0, Q, size=N))


def sample_ternary_poly(rng: np.random.Generator) -> Poly:
    # coefficients uniform in {-1, 0, 1}
    return Poly(rng.integers(-1, 2, size=N))


def sample_gauss_poly(rng: np.random.Generator, sigma: float, tail_cut: float = 9.0) -> Poly:
    # draw a discrete-Gaussian-ish polynomial via batched rejection sampling
    bound = int(np.ceil(tail_cut * sigma))
    out = np.empty(N, dtype=np.int64)
    filled = 0
    while filled < N:
        batch = (N - filled) * 3  # oversample so most calls finish in one pass
        candidates = rng.integers(-bound, bound + 1, size=batch)
        accept_prob = np.exp(-(candidates.astype(np.float64) ** 2) / (2.0 * sigma * sigma))
        accepted = candidates[rng.random(batch) < accept_prob]
        take = min(len(accepted), N - filled)
        out[filled:filled + take] = accepted[:take]
        filled += take
    return Poly(out)


def sample_uniform_vec(rng, m) -> list:
    # m independent uniform polynomials
    return [sample_uniform_poly(rng) for _ in range(m)]


def sample_ternary_vec(rng, m) -> list:
    # m independent ternary polynomials
    return [sample_ternary_poly(rng) for _ in range(m)]


def sample_gauss_vec(rng, m, sigma) -> list:
    # m independent Gaussian-ish polynomials
    return [sample_gauss_poly(rng, sigma) for _ in range(m)]


def expand_matrix(seed: bytes, rows: int, cols: int):
    # deterministically expand a seed into a public matrix (same seed -> same matrix)
    seed_int = int.from_bytes(hashlib.sha256(seed).digest()[:8], "big")
    rng = np.random.default_rng(seed_int)
    return [[sample_uniform_poly(rng) for _ in range(cols)] for _ in range(rows)]


def round_nu(v, nu: int):
    # modulus rounding: drop the low nu bits of each coefficient (nu=0 means "no dropping")
    if nu == 0:
        return vec_modq(v)
    q_nu = Q // (1 << nu)
    half = 1 << (nu - 1)
    out = []
    for p in v:
        x_bar = np.mod(p.c, Q)                  # canonical unsigned representative, [0, Q)
        x_top = (x_bar + half) // (1 << nu)      # round-half-up, divided by 2^nu
        x_top = np.mod(x_top, q_nu)              # land in Z_{q_nu}
        out.append(Poly(x_top))
    return out
