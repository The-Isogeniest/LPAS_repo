"""
Parameter sets for the LPAS proof-of-concept.

Two profiles are available, selected via the LPAS_PROFILE environment
variable (default "toy"). See demo.py's --profile flag, which is the
intended way to choose one:

  "toy"   Small, fast, INSECURE parameters for reading/running the demo
          quickly. Modulus-rounding/compression is disabled (nu_b=nu_w=0,
          see README.md "Simplifications"), so the hint h is always zero.

  "paper" The concrete parameter set from Table 2 / Appendix E ("Parameter
          Selection") of the paper, including real compression
          (nu_b=9, nu_w=15), so h is a genuine small nonzero hint here,
          not the trivial all-zero vector of the toy profile. Targets
          128-bit classical security per the paper's own estimate against
          best-known attacks on the resulting MLWE/MSIS instances (not by
          re-deriving the security-proof's loose reduction bounds, see the
          paper's Section 5). n=256 and the larger module rank make this
          noticeably slower than "toy", though still fast enough to run
          interactively.

Either way: this is a reading/demo implementation, not a security-reviewed
one (see README.md for the list of simplifications, e.g. the NIZK stub and
the rounded-normal approximation to a discrete Gaussian). Only the "paper"
profile's *sizes* come from the paper; the implementation around them
(NIZK stub, rounded-normal sampler, etc.) is unchanged.
"""
import os

# ---------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------

_TOY = dict(
    N=64, Q=3329, K=2, L=2,
    OMEGA=20,
    SIGMA_Y=40.0, SIGMA_Y_TILDE=40.0, SIGMA_WIT=40.0,
    REJ_M_Y=2.0, REJ_M_Y_TILDE=2.0, REJ_M_WIT=2.0,
    NU_B=0, NU_W=0,
    B_WIT=1200.0, B_PROXYSIGN=4000.0, B_SIGN=4000.0,
)

_PAPER_SIGMA_Y_TILDE = 2 ** 16.47
_PAPER_SIGMA_WIT = 2 ** 11.41

_PAPER = dict(
    # Table 2 / Appendix E ("Parameter Selection"): n, k, l, q, omega.
    N=256, Q=56430593, K=4, L=4,         # q = 2^9*110216+1 = 2^15*1722+4097, prime, q = 1 (mod 2n)
    OMEGA=60,
    # sigma~_w, sigma_wit (Appendix E derives these from M_s=M_ps=M_wit=1.3
    # and target Renyi-divergence errors eps_s=eps_ps<=2^-68, eps_wit<=2^-194).
    # sigma_w (ordinary signing) is then set to sqrt(sigma~_w^2 + sigma_wit^2),
    # since an adapted response is z = z~ + wit'.
    SIGMA_Y=(_PAPER_SIGMA_Y_TILDE ** 2 + _PAPER_SIGMA_WIT ** 2) ** 0.5,
    SIGMA_Y_TILDE=_PAPER_SIGMA_Y_TILDE, SIGMA_WIT=_PAPER_SIGMA_WIT,
    REJ_M_Y=1.3, REJ_M_Y_TILDE=1.3, REJ_M_WIT=1.3,
    # Modulus-rounding (Definition 1): real compression is ON here.
    NU_B=9, NU_W=15,
    # Norm bounds, Appendix E (Euclidean norm).
    B_WIT=2 ** 17.27, B_PROXYSIGN=2 ** 22.99, B_SIGN=2 ** 23.02,
)

_PROFILES = {"toy": _TOY, "paper": _PAPER}

PROFILE = os.environ.get("LPAS_PROFILE", "toy").lower()
if PROFILE not in _PROFILES:
    raise ValueError(f"unknown LPAS_PROFILE {PROFILE!r}; expected one of {sorted(_PROFILES)}")
_p = _PROFILES[PROFILE]

# ---- Ring R_q = Z_q[x] / (x^n + 1) --------------------------------------
N = _p["N"]             # ring degree (paper: n)
Q = _p["Q"]              # modulus (paper: q)

# ---- Module dimensions ---------------------------------------------------
K = _p["K"]              # rows of the public matrix (paper: k)
L = _p["L"]              # columns of the public matrix Abar (paper: l)
M = K + L                # length of secret/witness/randomness vectors (l+k)

# ---- Security parameter ---------------------------------------------------
# Fixed at 256 bits regardless of profile (matches AES-256's key size and
# our SHA-256-based H). Note N*M need not equal LAMBDA: pke.py embeds a
# LAMBDA-bit message into the first LAMBDA of the N*M available plaintext
# coefficient slots, zero-padding the rest when N*M > LAMBDA.
LAMBDA = 256

# ---- Signature scheme (Figure 2) distributions ---------------------------
NU_B = _p["NU_B"]        # bits dropped from the verification key / commitment
NU_W = _p["NU_W"]        # (Definition 1). 0 in the toy profile == disabled.

SIGMA_Y = _p["SIGMA_Y"]              # std-dev of ordinary signing randomness y   (chi_y)
SIGMA_Y_TILDE = _p["SIGMA_Y_TILDE"]  # std-dev of proxy-signing randomness y      (chi~_y)
SIGMA_WIT = _p["SIGMA_WIT"]          # std-dev of the blindness r                 (chi_wit)

REJ_M_Y = _p["REJ_M_Y"]              # rejection-sampling parameter M_s  (Sign)
REJ_M_Y_TILDE = _p["REJ_M_Y_TILDE"]  # rejection-sampling parameter M_ps (ProxySign)
REJ_M_WIT = _p["REJ_M_WIT"]          # rejection-sampling parameter M_wit (AdGen)

OMEGA = _p["OMEGA"]      # Hamming weight of the challenge polynomial c
                          # (paper: ||c||_1 = omega, ||c||_inf = 1)

MAX_REJ_TRIES = 500       # safety cap on rejection-sampling retries

# Norm bounds (Table 1/2, Section 3, Appendix E). Toy-profile values are
# generous placeholders (not derived from the Renyi-divergence analysis);
# paper-profile values are Appendix E's derived bounds.
B_SIGN = _p["B_SIGN"]
B_PROXYSIGN = _p["B_PROXYSIGN"]
B_WIT = _p["B_WIT"]
