# LPAS proof of concept

 **single-proxy LPAS**
construction (Figure 3) from the paper "LPAS: Lattice-based Proxy
Adaptor Signatures". 


This repository represents a proof-of-concept and it is not an optimized code. There are also some simplified
parts implemented as we mention later

## Requirements

- Python 3.10+
- `numpy`
- `cryptography`

```bash
pip install numpy cryptography
```

- `numpy`: array/math library, used here for random number generation (`SeedSequence`) and vector arithmetic.
- `cryptography`: general-purpose crypto library, used here only for AES-256-GCM (the SKE part of the scheme).

## Usage

```bash
python3 demo.py
```
runs one full exchange on the toy profile, with fresh randomness and full
step-by-step output. `demo.py` also takes four optional flags, which can
be combined freely:

| Flag | Default | Meaning |
|---|---|---|
| `--profile toy\|paper` | `toy` | Which parameter set to run with (see the table below). |
| `--seed N` | random (OS entropy) | Fixes the RNG so the run (or whole batch) is reproducible. |
| `--trials N` | `1` | Run the exchange N times instead of once. |
| `--verbose True\|False` | `True` | Full step-by-step detail per trial, or just a one-line pass/fail summary per trial. |

A few examples:

```bash
python3 demo.py --profile paper
```
Same single run, but on the paper's actual Table 2 / Appendix E parameters
(real modulus-rounding compression included), instead of the fast toy set.

```bash
python3 demo.py --seed 42
```
Reproducible run: the witness, challenge, blindness, etc. come out
identical every time you run this exact command.

```bash
python3 demo.py --trials 50
```
Stress test: runs the exchange 50 times. Trial 1 prints full detail, the
rest print a one-line pass/fail summary, and a batch summary ("X/N
succeeded", timing min/avg/max) is printed at the end.

```bash
python3 demo.py --profile paper --trials 30
```
Same stress test, on the real parameters.

```bash
python3 demo.py --trials 5 --verbose True
```
Forces every trial to print full detail, not just the first one.

```bash
python3 demo.py --verbose False
```
Forces a concise one-line summary even for a single run, no step-by-step
output at all.

```bash
python3 demo.py --profile paper --trials 100 --verbose False --seed 7
```
A quiet, reproducible 100-trial batch on the real parameters, just the
pass/fail summary for each one plus the final aggregate.

```bash
python3 demo.py --help
```
Lists every flag and its default directly from the CLI.



## Parameter profiles

Two parameter profiles are available (`params.py`, selected via `--profile` /
the `LPAS_PROFILE` env var):

| | `toy` (default) | `paper` |
|---|---|---|
| `n`, `q` | 64, 3329 | 256, 56430593 |
| `k`, `l` | 2, 2 | 4, 4 |
| `ω` (challenge weight) | 20 | 60 |
| compression `ν_b`, `ν_w` | 0, 0 (disabled) | 9, 15 (real) |
| runtime (full exchange) | ~8 ms | ~56 ms |



## Sizes (paper profile)


Below are the resulting sizes,which match the paper's Table 2 / Appendix E
exactly, plus a short note on how each is packed.

| Component | Size | How it's computed |
|---|---|---|
| `vk` | 2.14 KiB | seed (16 B) + `b`'s `nk` coefficients packed at the compressed modulus `q_b = q/2^ν_b`, `⌈log₂ q_b⌉` bits each. |
| `req` | 4.50 KiB | `PKE.pk` + `stmt`'s `k` coefficients packed at the full modulus `q`. |
| `t` | 3.25 KiB | same packing as `stmt`, at modulus `q`. |
| `PKE.pk` | 1.25 KiB | seed (32 B) + `b`'s `D=3` coefficients packed at the PKE's own modulus `q_PKE=7681`. |
| `PKE.ct` | 1.625 KiB | `c1` (`D` ring elements, `Dn` coefficients) + `c2` (1 ring element, `n` coefficients), all mod `q_PKE`, summed over however many chunks the message needs. |
| `sig` | 6.53 KiB | challenge `c` (sparse rank encoding: `⌈log₂ C(n,ω)⌉ + ω` bits) + `z`'s `n(k+l)` coefficients at a fixed width sized to the public bound `B_sign` + `h`'s `nk` coefficients at 2 bits (paper shows `h ∈ {-1,0,1}` for these parameters). |
| `presig` | 6.28 KiB | same as `sig`, but `z~`'s width is sized to `B_proxysign` instead. |
| `advt` (excl. NIZK) | 9.65 KiB | `t` + `PKE.ct` + `SKE.ct` (AES-256-GCM of `r`, tight-packed to a fixed width sized to `B_wit`, +12 B nonce +16 B tag). |
| `(wit', ct)` to buyer | 11.15 KiB | `wit'`'s `n(k+l)` coefficients at the same `B_wit`-sized width + `PKE.ct` + `SKE.ct` (same ciphertext, forwarded from `advt`). |



## Files

| File | Contents |
|---|---|
| `params.py` | Two parameter profiles, `toy` and `paper` (see above). Both are for reading/running this demo, not production use. |
| `ring.py` | `R = Z[x]/(x^n+1)` arithmetic, module vectors/matrices, sampling, modulus rounding, and the compact wire-packing helpers behind the sizes above (fixed-width bit-packing, mod-`q` packing, sparse-challenge rank encoding). |
| `signature.py` | The underlying signature scheme `Pi` = (KeyGen, Sign, Verify), Figure 2 (Fiat–Shamir with aborts), plus `Rej`, Figure 4. |
| `pke.py` | The MLWE-based PKE of Figure 11 (Appendix D), matching the paper's KeyGen/Enc/Dec and its own `q_PKE=7681`, `d=3` (Appendix E); reuses the active profile's ring degree `n` instead of the paper's fixed `n=256`. |
| `ske.py` | AES-256-GCM, the paper's stated SKE instantiation. |
| `nizk.py` | **Stub**, not a real proof system |
| `lpas.py` | Figure 3 itself: `Setup, ReqGen, ReqVerify, AdGen, AdVerify, ProxySign, ProxyPreVerify, Adapt, ProxyExt, ReqExt`, and the hard relation `R_A`. |
| `demo.py` | End-to-end run: Buyer/Seller/Proxy simulation with printed steps and assertions. |


## Suggested next steps

- Replace `nizk.py` with a real Fiat-Shamir-compiled Sigma-protocol for statement (7)-(8) (a linear-relation proof over `R_A` combined with an encryption-correctness proof for the PKE/SKE ciphertexts).
- For now, the `nizk.py` file does not implement the full NIZK proofs.
