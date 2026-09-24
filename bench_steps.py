"""Per-algorithm timing benchmark for LPAS (paper profile), mirroring the
exact call sequence of demo.py's run_trial, but collecting every step's
wall-clock time across many trials instead of printing one run.
"""
import os
import sys
import time
import statistics as st
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--profile", choices=["toy", "paper"], default="paper")
parser.add_argument("--trials", type=int, default=500)
parser.add_argument("--seed", type=int, default=123)
args = parser.parse_args()

os.environ["LPAS_PROFILE"] = args.profile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import params as P
import lpas
import signature as SIG

STEPS = ["Setup", "ReqGen", "ReqVerify", "AdGen", "AdVerify", "ProxySign",
         "ProxyPreVerify", "Adapt", "Pi.Verify", "ProxyExt", "ReqExt"]
times = {s: [] for s in STEPS}

def timed(label, fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    times[label].append((time.perf_counter() - t0) * 1000.0)
    return out

seed_seq = np.random.SeedSequence(args.seed)
child_seeds = seed_seq.spawn(args.trials)

ok = 0
for i, child_seed in enumerate(child_seeds):
    rng = np.random.default_rng(child_seed)
    pp, proxy_sk = timed("Setup", lpas.setup, rng)
    stmt, wit = lpas.gen_relation_A(rng, pp.A)  # test-only helper, not timed as a protocol step
    req, st_buyer = timed("ReqGen", lpas.req_gen, rng, stmt)
    assert timed("ReqVerify", lpas.req_verify, stmt, req)
    advt, st_advt = timed("AdGen", lpas.ad_gen, rng, pp, req, wit)
    assert timed("AdVerify", lpas.ad_verify, pp, req, advt)
    msg = b"pay 0.10 BTC to seller for report #1337"
    presig = timed("ProxySign", lpas.proxy_sign, rng, pp, proxy_sk, msg, req, advt)
    assert timed("ProxyPreVerify", lpas.proxy_pre_verify, pp, msg, req, advt, presig)
    sig_full = timed("Adapt", lpas.adapt, pp, msg, req, advt, presig, wit, st_advt)
    assert timed("Pi.Verify", SIG.verify, pp.vk, pp.A, msg, sig_full)
    wit_blinded = timed("ProxyExt", lpas.proxy_ext, sig_full, presig)
    wit_recovered = timed("ReqExt", lpas.req_ext, pp, wit_blinded, st_buyer, stmt, advt)
    if wit_recovered is not None and all(a == b for a, b in zip(wit_recovered, wit)):
        ok += 1
    times.setdefault("TOTAL", []).append(sum(times[s][-1] for s in STEPS))
    fig3_steps = [s for s in STEPS if s != "Pi.Verify"]
    times.setdefault("TOTAL_FIG3", []).append(sum(times[s][-1] for s in fig3_steps))

print(f"profile={args.profile}  trials={args.trials}  ok={ok}/{args.trials}\n")
header = f"{'step':<16}{'n':>6}{'min':>10}{'median':>10}{'mean':>10}{'stdev':>10}{'max':>10}"
print(header)
print("-" * len(header))
for s in STEPS:
    v = times[s]
    print(f"{s:<16}{len(v):>6}{min(v):>10.3f}{st.median(v):>10.3f}{st.mean(v):>10.3f}{st.stdev(v):>10.3f}{max(v):>10.3f}")
print("-" * len(header))
v = times["TOTAL"]
print(f"{'TOTAL (w/ Pi.V)':<16}{len(v):>6}{min(v):>10.3f}{st.median(v):>10.3f}{st.mean(v):>10.3f}{st.stdev(v):>10.3f}{max(v):>10.3f}")
v = times["TOTAL_FIG3"]
print(f"{'TOTAL (Fig.3)':<16}{len(v):>6}{min(v):>10.3f}{st.median(v):>10.3f}{st.mean(v):>10.3f}{st.stdev(v):>10.3f}{max(v):>10.3f}")
