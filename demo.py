"""
End-to-end proof-of-concept run of the single-proxy LPAS scheme (Figure 3).

A Seller holds a witness `wit` for a public statement `stmt` (e.g. a
secret unlocking some digital asset). A Buyer wants that asset and delegates
the exchange to a Proxy holding a signing key for some transaction `msg`
(e.g. a payment). The Proxy can complete the exchange without ever learning
`wit`, and the Buyer recovers `wit` only from the public artefacts left
behind once the exchange completes.

Every value of interest is shown as hex and every milestone is timed.

Two parameter profiles are available (see params.py): --profile toy
(default, small/fast/insecure) or --profile paper (the paper's real
parameters, with real compression). Pass --seed for a reproducible run,
--trials N to repeat it, and --verbose False for a quiet pass/fail summary.

Run with:  python3 demo.py [--profile toy|paper] [--seed N] [--trials N] [--verbose True|False]
"""
import argparse
import os

def _bool(v: str) -> bool:
    # parse a True/False CLI flag
    if v.lower() in ("true", "1"):
        return True
    if v.lower() in ("false", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected True or False, got {v!r}")


# Parse --profile and set LPAS_PROFILE before importing any project module:
# params.py reads that env var once, at import time.
_parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
_parser.add_argument("--profile", choices=["toy", "paper"], default="toy",
                      help="parameter profile to run with (default: toy)")
_parser.add_argument("--seed", type=int, default=None,
                      help="RNG seed for a reproducible run/batch (default: random each run)")
_parser.add_argument("--trials", type=int, default=1,
                      help="number of independent exchanges to run (default: 1)")
_parser.add_argument("--verbose", type=_bool, default=True,
                      help="True (default): print full detail for every trial. False: print "
                           "only a one-line pass/fail summary per trial, uniformly, with no "
                           "special-casing of any particular trial.")
_args = _parser.parse_args()
if _args.trials < 1:
    _parser.error("--trials must be >= 1")
os.environ["LPAS_PROFILE"] = _args.profile

import time  # noqa: E402  (must follow the LPAS_PROFILE env-var setup above)

import numpy as np  # noqa: E402

import params as P  # noqa: E402
import lpas  # noqa: E402
import pke as PKE  # noqa: E402
import signature as SIG  # noqa: E402
from ring import vec_l2, pack_ternary, pack_bits, pack_modq, pack_challenge, best_group_size, bits_needed  # noqa: E402

# MODQ_GROUP: how many mod-q coefficients to pack together per hex dump group
# (avoids wasting bits when q isn't a power of two).
MODQ_GROUP = best_group_size(P.Q)

# How much of a packed value's hex to actually print (data itself stays full/lossless).
HEX_PREVIEW_BYTES = 48


def _hex_display(packed: bytes) -> str:
    # truncate only the printed hex, not the underlying data
    if len(packed) <= HEX_PREVIEW_BYTES:
        return packed.hex()
    shown = packed[:HEX_PREVIEW_BYTES].hex()
    return f"{shown}...  (+{len(packed) - HEX_PREVIEW_BYTES} more bytes not shown)"


def hexbytes(data: bytes) -> str:
    # full hex dump, for already-opaque byte strings (keys, hashes, ciphertexts)
    return data.hex()


def raw_prefix(data: bytes, n: int = 16) -> str:
    # hex prefix + total length, for long opaque byte strings
    return f"{data[:n].hex()}...  ({len(data)} bytes total)"


def flat_coeffs(v) -> list:
    # flatten a vector of Poly into a plain list of ints
    return [int(x) for p in v for x in p.c]


def thex(coeffs, preview_n: int = 10) -> str:
    # compact lossless hex for ternary ({-1,0,1}) coefficients
    packed = pack_ternary(coeffs)
    preview = coeffs[:preview_n]
    return (f"{_hex_display(packed)}  "
            f"({len(coeffs)} ternary coeffs packed 5/byte -> {len(packed)} bytes; "
            f"first {preview_n} = {preview})")


def bhex(coeffs, bit_width: int, preview_n: int = 10) -> str:
    # compact lossless hex for coefficients known to fit in bit_width bits
    packed = pack_bits(coeffs, bit_width)
    preview = coeffs[:preview_n]
    return (f"{_hex_display(packed)}  "
            f"({len(coeffs)} coeffs packed @ {bit_width} bits/coeff -> {len(packed)} bytes; "
            f"first {preview_n} = {preview})")


def phex(p, preview_n: int = 16) -> str:
    # compact lossless hex for the sparse challenge polynomial c: encodes
    # (which OMEGA positions are nonzero, their signs) via pack_challenge,
    # not a trit per coefficient -- matches the paper's Appendix E sizing
    # (|c| = ceil(log2 C(n,omega)) + omega bits), unlike thex()'s generic,
    # sparsity-unaware ternary packing.
    coeffs = [int(x) for x in p.c]
    packed = pack_challenge(coeffs, P.OMEGA)
    preview = coeffs[:preview_n]
    return (f"{_hex_display(packed)}  "
            f"({P.N} coeffs, {P.OMEGA} nonzero, packed as (support rank, signs) "
            f"-> {len(packed)} bytes; first {preview_n} = {preview})")


def pubhex(v, q: int = None, preview_n: int = 10) -> str:
    # compact lossless hex for a public mod-q value (vector or matrix of Poly).
    # defaults to the signature scheme's modulus P.Q; pass q=PKE.Q_PKE for
    # PKE values, which live at their own, separate modulus.
    if q is None:
        q = P.Q
    group = MODQ_GROUP if q == P.Q else best_group_size(q)
    polys = [p for row in v for p in row] if isinstance(v[0], list) else v
    flat = flat_coeffs(polys)
    packed = pack_modq(flat, q, group)
    preview = [c % q for c in flat[:preview_n]]
    return (f"{_hex_display(packed)}  "
            f"({len(flat)} coeffs packed @ {group}/group, "
            f"{8 * len(packed) / len(flat):.1f} bits/coeff -> {len(packed)} bytes; "
            f"first {preview_n} (mod q) = {preview})")


def shorthex(v, bit_width: int) -> str:
    # compact lossless hex for a short Gaussian-ish vector, packed at a FIXED,
    # publicly-known bit-width (see Z_BITS/ZT_BITS/WIT_BITS/H_BITS below), not
    # one measured from this run's own coefficients. A width picked from the
    # actual data can't be unpacked by a real receiver, who sees only bytes
    # and has no way to learn, ahead of time, how wide THIS run's values
    # happened to be -- a usable wire format has to fix the width in advance,
    # from a bound both sides already know, exactly as the paper's own
    # sizing (Appendix E) does. pack_bits() asserts loudly if some value
    # ever exceeds this width, which would mean the norm bound was violated.
    return bhex(flat_coeffs(v), bit_width)


# Fixed per-coefficient bit-widths for "short" vectors, derived ONCE from
# each vector's PUBLIC norm bound (params.py), not sampled per run -- see
# shorthex() above for why. Matches the paper's own Appendix E accounting:
# Z_BITS=25, ZT_BITS=24 for the paper profile (since B_sign<2^24 resp.
# B_proxysign<2^23, each needs one extra bit for the sign).
Z_BITS = bits_needed(int(np.ceil(P.B_SIGN)))
ZT_BITS = bits_needed(int(np.ceil(P.B_PROXYSIGN)))
WIT_BITS = lpas.WIT_BITS  # same width ad_gen/req_ext actually pack r/wit' at
if P.NU_B == 0 and P.NU_W == 0:
    H_BITS = 1  # compression disabled: h is always the all-zero vector
elif P.OMEGA * (2 ** P.NU_B - 1) < 2 ** P.NU_W:
    H_BITS = 2  # paper's Appendix E "Sizes": h's coefficients land in {-1,0,1}
else:
    H_BITS = bits_needed(2 ** P.NU_B)  # conservative fallback outside that case

# vk.b (and any other value from round_nu(., NU_B)) lives mod q_b, the
# COMPRESSED modulus -- not the full Q -- so it must be packed at q_b too,
# or the printed size overstates it (same "pack at the value's own live
# modulus" principle as pubhex's q parameter).
Q_B = P.Q // (1 << P.NU_B) if P.NU_B > 0 else P.Q


class Timer:
    """Times each milestone in isolation and, when verbose, prints a running total."""

    def __init__(self, verbose: bool = True):
        self.t0 = time.perf_counter()
        self.verbose = verbose

    def time_call(self, label: str, fn, *args, **kwargs):
        # time just this one call, not anything printed before/after it
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        if self.verbose:
            step_ms = (time.perf_counter() - start) * 1000
            total_ms = (time.perf_counter() - self.t0) * 1000
            print(f"    [timing] {label}: {step_ms:.2f} ms  (running total: {total_ms:.2f} ms)")
        return result

    @property
    def total_ms(self) -> float:
        # elapsed time since the timer was created
        return (time.perf_counter() - self.t0) * 1000


def run_trial(rng: np.random.Generator, verbose: bool) -> dict:
    # run one full LPAS exchange (Setup through ReqExt) and report success/failure
    out = print if verbose else (lambda *a, **k: None)

    def section(title):
        out("\n" + "=" * 78)
        out(title)
        out("=" * 78)

    timer = Timer(verbose)
    try:
        compression_note = ("disabled (nu_b=nu_w=0), h below will be all-zero"
                             if P.NU_B == 0 and P.NU_W == 0 else
                             f"ENABLED (nu_b={P.NU_B}, nu_w={P.NU_W}), h below is a genuine small hint")

        section(f"Trusted setup (profile={P.PROFILE!r}; creates the Proxy's signing key)")
        pp, proxy_sk = timer.time_call("Setup", lpas.setup, rng)
        out(f"ring: n={P.N}, q={P.Q}, module rank l+k={P.M}   [INSECURE demo parameters]")
        out(f"compression: {compression_note}")
        out(f"vk.seed  = {hexbytes(pp.vk[0])}")
        out(f"vk.b     = {pubhex(pp.vk[1], Q_B)}")
        out("Proxy verification key vk = (seed, b) published; Proxy alone holds sk.")

        section("Seller's asset: a witness wit for a public statement stmt")
        stmt, wit = timer.time_call("gen_relation_A (test-only helper)", lpas.gen_relation_A, rng, pp.A)
        out(f"stmt = A . wit  = {pubhex(stmt)}")
        out(f"wit (ternary)   = {thex(flat_coeffs(wit))}")

        section("1) Buyer, ReqGen: request to buy the asset behind `stmt`")
        req, st_buyer = timer.time_call("ReqGen", lpas.req_gen, rng, stmt)
        pke_seed, pke_b = req[0]
        out(f"PKE.pk.seed = {hexbytes(pke_seed)}")
        out(f"PKE.pk.b    = {pubhex(pke_b, PKE.Q_PKE)}")
        out("Buyer generated a fresh PKE keypair and req = (PKE.pk, stmt); sent to Proxy.")

        section("2) Proxy, ReqVerify")
        assert timer.time_call("ReqVerify", lpas.req_verify, stmt, req)
        out("Proxy accepted the request and published it on the bulletin board.")

        section("3) Seller, AdGen: advertisement, blinding the witness")
        advt, st_advt = timer.time_call("AdGen", lpas.ad_gen, rng, pp, req, wit)
        pi_advt, t, (ct_pke, ct_ske) = advt
        pke_c1_polys = [poly for (c1, c2) in ct_pke for poly in c1]
        pke_c2_polys = [c2 for (c1, c2) in ct_pke]
        r = st_advt
        out(f"r (blindness)      = {shorthex(r, WIT_BITS)}")
        out(f"t = A . r          = {pubhex(t)}")
        out(f"ct.c1 (PKE, part1) = {pubhex(pke_c1_polys, PKE.Q_PKE)}")
        out(f"ct.c1 (PKE, part2) = {pubhex(pke_c2_polys, PKE.Q_PKE)}")
        out(f"  ({len(ct_pke)} independent PKE ciphertext chunk(s) of {P.N} bits each, "
            f"sharing one PKE keypair)")
        out(f"ct.c2 (AES-256-GCM)= {raw_prefix(ct_ske)}")
        out("Seller sampled blindness r, computed wit' = wit + r (accepted by rejection")
        out("sampling), encrypted a one-time key under the Buyer's PKE key, and")
        out("encrypted r under that key via AES-256-GCM. advt sent to Proxy.")

        section("4) Proxy, AdVerify")
        assert timer.time_call("AdVerify", lpas.ad_verify, pp, req, advt)
        out("Proxy accepted the advertisement (NIZK stub check passed).")

        msg = b"pay 0.10 BTC to seller for report #1337"
        out(f"\nmsg (hex) = {hexbytes(msg)}   ({msg!r})")

        section("5) Proxy, ProxySign: pre-signature on msg, shifted by stmt+t")
        presig = timer.time_call("ProxySign", lpas.proxy_sign, rng, pp, proxy_sk, msg, req, advt)
        c_tilde, z_tilde, h_tilde = presig
        out(f"c~ (challenge)  = {phex(c_tilde)}")
        out(f"z~ (response)   = {shorthex(z_tilde, ZT_BITS)}   (||z~||_2 = {vec_l2(z_tilde):.1f})")
        out(f"h~ (hint)       = {shorthex(h_tilde, H_BITS)}")

        section("6) Seller, ProxyPreVerify + Adapt")
        assert timer.time_call("ProxyPreVerify", lpas.proxy_pre_verify, pp, msg, req, advt, presig)
        out("Seller verified the pre-signature, then adapted it using wit:")
        sig_full = timer.time_call("Adapt", lpas.adapt, pp, msg, req, advt, presig, wit, st_advt)
        c_sig, z_sig, h_sig = sig_full
        out(f"z (full response) = {shorthex(z_sig, Z_BITS)}   (||z||_2 = {vec_l2(z_sig):.1f})")
        out("Full, spendable signature produced. This is the irreversible step: the seller")
        out("has now committed wit into sigma_sig (ProxyExt can recover wit' from it whether")
        out("or not it's ever broadcast). Actually broadcasting it to a real network and")
        out("getting it confirmed, i.e. the payment actually settling, is outside what this")
        out("demo simulates; only the signing/extraction cryptography is modeled here.")

        section("Check: the adapted signature verifies as an ordinary Pi signature")
        ok = timer.time_call("Pi.Verify", SIG.verify, pp.vk, pp.A, msg, sig_full)
        out(f"Pi.Verify(vk, msg, sigma_sig) = {ok}")
        assert ok, "adapted signature failed to verify!"

        section("7) Proxy, ProxyExt: extract the blinded witness from (sigma_sig, sigma~_sig)")
        wit_blinded = timer.time_call("ProxyExt", lpas.proxy_ext, sig_full, presig)
        out(f"wit' = z - z~  = {shorthex(wit_blinded, WIT_BITS)}   (||wit'||_2 = {vec_l2(wit_blinded):.1f})")
        out("Proxy sends (wit', advt.ct) to the Buyer. Proxy never sees `wit` or `r`.")

        section("8) Buyer, ReqExt: recover the original witness")
        wit_recovered = timer.time_call("ReqExt", lpas.req_ext, pp, wit_blinded, st_buyer, stmt, advt)
        assert wit_recovered is not None, "ReqExt failed to recover a valid witness!"
        match = all(a == b for a, b in zip(wit_recovered, wit))
        out(f"wit* (recovered) = {thex(flat_coeffs(wit_recovered))}")
        out(f"wit  (original)  = {thex(flat_coeffs(wit))}")
        out(f"wit* == original wit ?  {match}")
        assert match, "recovered witness does not match the seller's original witness!"

        section("Result")
        out(f"End-to-end run succeeded (profile={P.PROFILE!r}): extractability (Proxy gets")
        out("wit') and the buyer-side recovery (Buyer alone gets wit) both hold, and the")
        out("adapted signature verifies as an ordinary signature on `msg`.")
        out(f"\nTotal wall-clock time for the whole exchange: {timer.total_ms:.2f} ms")
        return {"ok": True, "total_ms": timer.total_ms, "error": None}

    except Exception as e:  # noqa: BLE001 (deliberately broad: any failure ends this trial, not the batch)
        out(f"\n*** TRIAL FAILED: {type(e).__name__}: {e} ***")
        return {"ok": False, "total_ms": timer.total_ms, "error": f"{type(e).__name__}: {e}"}


def main():
    # run one or more trials per the CLI flags, then print a batch summary if there's more than one
    trials = _args.trials
    seed_seq = np.random.SeedSequence(_args.seed)
    child_seeds = seed_seq.spawn(trials)
    seed_desc = _args.seed if _args.seed is not None else "random (OS entropy), rerun with --seed N to reproduce"
    print(f"RNG seed: {seed_desc}" + (f"   ({trials} trials)" if trials > 1 else ""))

    verbose = _args.verbose

    results = []
    for i, child_seed in enumerate(child_seeds):
        rng = np.random.default_rng(child_seed)
        if trials > 1:
            print(f"\n--- trial {i + 1}/{trials} ---")
        result = run_trial(rng, verbose)
        results.append(result)
        if not verbose:
            status = "OK" if result["ok"] else f"FAILED ({result['error']})"
            print(f"trial {i + 1}/{trials}: {status}  ({result['total_ms']:.2f} ms)")

    if trials > 1:
        ok_count = sum(r["ok"] for r in results)
        times = [r["total_ms"] for r in results]
        print("\n" + "=" * 78)
        print(f"Batch summary: {ok_count}/{trials} trials succeeded")
        print(f"Timing (ms): min={min(times):.2f}  avg={sum(times) / len(times):.2f}  max={max(times):.2f}")
        if ok_count < trials:
            print("Failed trials:")
            for i, r in enumerate(results):
                if not r["ok"]:
                    print(f"  #{i + 1}: {r['error']}")
        print("=" * 78)


if __name__ == "__main__":
    main()
