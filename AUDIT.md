# Critical audit — Julia-native checker (`julia_poc/` ≡ DB_Julia)

An independent, deliberately critical look at the Julia detailed-balance
checker, and a reconciliation with the two design notes (`julia_bfs.md`,
`optimisation.md`) written before it existed. Written 2026-06-04.

---

## 1. What the prior design notes got right, wrong, or now-outdated

### `julia_bfs.md` (τ-BFS feasibility PoC)

- **Right:** the core thesis — Julia can replicate the τ-BFS exactly with
  `TauPos` (→ `TauNum`), a `BitSeqRNG`, and `DeltaE` vectors, with no symbolic
  algebra library — is borne out. The "always-read policy", the τ=0 leaf
  substitution, and the τ-violation-via-explicit-tracking all made it into the
  production code.
- **Understated the threshold problem.** The PoC only ever handled
  Metropolis-style *clamped* thresholds `min(1, exp(−βΔE))`. VMMC needs far
  more: a **ratio of `(1−exp)` sums capped by `Min`**, which does not reduce to
  a sum of exponentials until two acceptance factors are multiplied. None of
  this is anticipated in `julia_bfs.md`. It required a whole extra component —
  the exact rational-function engine (`Val = num / ∏(1−exp) binomials`) with
  denominator clearing — that the note's "DeltaE-vector storage" sketch does
  not cover. This was the single largest piece of unforeseen work.
- **Contains a latent bug it did not catch.** The PoC's `rand_choice!` used
  `k = ceil(log2(n + 0.5))`, which is **wrong for powers of two** (`n=8` →
  `k=4` instead of 3) and would have mis-weighted `broken_8way_hop`. The
  production code uses `nbits(n) = ndigits(n-1; base=2)` (Mathematica's
  `IntegerLength[n-1,2]`). Worth flagging that the PoC "passed" only because it
  never exercised a power-of-two pool.
- **Effort estimate** ("vmmc_2d: 2–4 days") was in the right ballpark.

### `optimisation.md` (Mathematica + Julia Phase 2/3)

- **Accurate** for the architecture it describes (compact-JSON export +
  `ebe.jl` subprocess, non-strict BFS + degenerate filter). The Julia-native
  checker reproduces that Phase-2 logic faithfully and finds the **same 216
  genuine chambers** for VMMC and **48** for single-Metropolis.
- **Now falsified:** "The Mathematica floor (τ-BFS + ergodicity ≈ 16 s) cannot
  be reduced by Julia." The Julia-native τ-BFS *with translation-orbit
  reduction* does the whole VMMC τ-BFS over the 56 orbit reps in ~16 s and the
  rest of the pipeline in ~10 s — i.e. the "floor" it called irreducible is now
  the entire runtime, and is itself reducible further (see §3).
- **Superseded:** the JSON round-trip, the `feasible_sigmas` export, the
  fallback-to-Mathematica path — all unnecessary in the native pipeline. The
  "PackageCompiler sysimage → 0.3 s" observation still stands as the obvious way
  to remove JIT warmup.

---

## 2. Soundness — where the checker is exact, and where it is not

The pipeline is exact (`Rational{BigInt}` throughout) **except** at the points
below. None causes a wrong verdict on the nine examples, but an honest user
should know the boundaries.

1. **τ-detection is sound only modulo the translation discipline.** Mathematica
   is *genuinely* symbolic: τ propagates through every expression, so any
   τ-dependence in any weight is detected automatically. The Julia checker
   detects τ-dependence only for values that flow through `TauNum` arithmetic
   and the provided geometry helpers. A translation that extracts `tau0(p.r)`
   early and branches on the raw integer would hide that dependence → a **false
   τ-PASS**. This is documented as a contract ("positions stay `TauNum`"), and
   it does not affect the *detailed-balance* verdict (which is computed at τ=0
   regardless), but it is a real weakening relative to Mathematica and is the
   checker's most important caveat.

2. **The `Min`-condition derivation makes an unasserted positivity assumption.**
   `min_condition` reduces `Min[a,b]`'s switch to a linear hyperplane by
   clearing the operands' denominators — which is only direction-preserving if
   those `(1−exp(−βL))` denominators are **positive** in the region where the
   `Min` is reached. For VMMC this holds (the `Min` sits under the guard
   `eInit<eFwd`, i.e. `L>0`), but the code **checks the balanced-binomial
   structure, not the positivity**. A `Min` outside the VMMC pattern could in
   principle get a mis-oriented hyperplane. This is bounded (only `ThMin`, only
   with denominators) and should either be asserted or documented as a hard
   limit; right now it is an implicit assumption.

3. **Chamber enumeration uses a floating-point LP (HiGHS, `eps=1e-6`).** This is
   the *only* non-exact step. The initial witness `J*_a = 100^a` is exact, and
   the hyperplanes have small integer coefficients, so mis-classifying a chamber
   would require a feasibility error on a chamber thinner than `eps` — not
   possible for these arrangements, but a theoretical gap. A rational LP /
   Fourier–Motzkin step would close it at a performance cost.

4. **Open chambers only.** Detailed balance is verified on the open chambers of
   the hyperplane arrangement; measure-zero boundaries (e.g. `J1=J2` exactly)
   are not separately checked. Sound because transition probabilities are
   continuous in the couplings, so any positive-measure violation appears in an
   adjacent open chamber. Inherited from the Mathematica design; correct but
   worth stating.

5. **Supported exact-weight class is bounded.** Weights must be rational
   functions whose denominators are products of `(1−exp(−βL))` binomials.
   Anything else (a different threshold algebra) is a **hard error**, not a
   guess — which is the right failure mode, but means "faithful translation" is
   only available for algorithms in this class.

---

## 3. Performance — honest state and the remaining headroom

VMMC went from **355 s → ~26 s** via four profiling-driven changes (per-pair
condition projection; lazy weight caching + unique-weight scan; translation-
orbit reduction; threshold hash-consing). The simple examples run in 6–11 s,
**most of which is Julia JIT warmup** (~4–5 s per process).

Remaining, *not* done (correctness was prioritised over these):

- **Lattice coordinates are carried as `Rational{BigInt}`.** Positions and
  distances are tiny integers; the residual VMMC bottleneck is the symbolic BFS,
  dominated by this. A narrower type (`Rational{Int64}`, or an integer position
  with a small τ-tracker) is the obvious next win — deferred only because it
  touches the foundational numeric type and a silent overflow there would be a
  wrong verdict, so it needs overflow-checked arithmetic to stay sound.
- **No memoisation of structurally-identical cluster sub-trees** in the VMMC
  BFS; the same partial cluster is re-expanded along different paths.
- **No PackageCompiler sysimage**, so every CLI invocation pays JIT. This is the
  difference between "≈10 s" and "≈1 s" for the simple examples.

The user's suggested "merge Phase 1 and Phase 2 (resolve inequalities during the
BFS)" was considered and **rejected on profiling grounds**: it would re-run the
BFS once per chamber (216× for VMMC), whereas the chosen design BFSs once and
evaluates thresholds per chamber lazily — strictly less work given that control
flow here is coupling-independent (only the *weights*, never the branch
structure, depend on the couplings).

---

## 4. Smaller criticisms

- **Determinism:** confirmed — no RNG anywhere; BFS over fixed bit trees;
  canonical `sort` on atoms/conditions; HiGHS LP is deterministic. Dict
  iteration order is used only to build order-independent sets. Good.
- **Hash-consing relies on `objectid` stability**, which holds because
  `_TH_CACHE` keeps interned nodes alive for the run and is reset at the start
  of `build_transitions`. Calling the `th_*` builders outside that lifecycle
  would be unsafe; this is an internal invariant, not part of the public API,
  but is undocumented in the code.
- **`enumerate_states` is O(P(S,N))** (all k-permutations) — fine for the small
  systems these checks target, exponential beyond them. The whole approach is
  exhaustive by design and does not scale to large lattices; it is a *verifier*
  for small representative systems, not a production sampler check.
- **Error messages are good** (`CantHandle` carries a specific reason), but a
  few internal `cant(...)` sites ("bad binomial numerator") would be opaque to a
  user translating a new algorithm.
- The SZ-DBC repo now carries **two engines** (`dbc_core.wl` Mathematica +
  `julia_poc/dbc.jl`); they are independent and must be kept in sync by hand if
  the semantics change. DB_Julia is the single-engine version.

---

## 5. Resolution log — 2026-06-04 (DB_Julia revision)

This revision was a focused pass to make the checker (a) fully trustworthy — no
floating point and no non-determinism anywhere in the verdict — (b) faster, and
(c) more general, with a clean template and a stress suite. Below, each issue
above (and several found during the work) is matched to what was done.

### 5.1 The floating-point LP is gone (was §2.3, the only non-exact step)

Chamber feasibility was the **only** place a float entered the verdict (HiGHS
with a `1e-6` tolerance). It is now decided by an **exact two-phase rational
simplex** with Bland's rule (`simplex_max` / `_is_feasible` in `dbc.jl`), so:

- there is no tolerance and no rounding anywhere in the pipeline;
- the heavy HiGHS binary dependency is removed (faster start-up, runs in more
  environments — addressing the "usable freely" goal);
- the result was **validated to agree with HiGHS on every sign pattern of every
  bundled example** before the switch (e.g. all 2047 patterns visited for VMMC,
  all 1931 for single-Metropolis — zero disagreements), and the chamber counts
  (48, 6, 216, …) are unchanged.

The exactness is principled, not just "agrees on these cases": the chamber-
adjacency graph is connected, so the sign-flip BFS reaches every chamber *given*
an exact feasibility oracle, and the simplex is that oracle.

### 5.2 Exact arithmetic is now also fast and overflow-sound (was §3, §2 caveat)

`Q` changed from `Rational{BigInt}` to **`Rational{Int128}`**. The BFS was
spending essentially all its time allocating GMP bignums for tiny-integer
operations; Int128 is native and removed that, giving a **~3× BFS speed-up** (VMMC
warm BFS 9.8 s → ~3 s; ~1 s with `-parallel`). Crucially this is *sound, not a
gamble*: Julia's `Rational{Int128}` arithmetic is **overflow-checked** and throws
`OverflowError` on overflow (verified), which `check.jl` catches and reports as a
hard error — never a silent wrong value. For the targeted system sizes overflow
does not occur; the witness power `base^a` is also chosen adaptively so it cannot
overflow for larger atom counts.

### 5.3 τ-detection is now sound for the orbit-reduction optimisation (NEW — a real latent bug)

§2.1 noted that τ-detection is "sound only modulo the translation discipline" and
claimed it "does not affect the detailed-balance verdict." **That claim was too
strong, and we found a concrete counterexample.** The orbit-reduction speed-up
trusts the τ flag: when it believes the algorithm is translation-invariant it
BFSs one representative per orbit and *translates* its leaves to the rest. If a
non-equivariant move slips past τ-detection, those translated leaves are wrong and
the DB verdict can be wrong.

The old detector only flagged τ-dependence that flowed through `pmod` / threshold
builders. A move that writes an **absolute or reflected next-state position**
without touching those (e.g. `Particle(-p.r, p.c, t)`, a row reflection) was *not*
flagged. We added `examples/reflect_move.jl` for exactly this case and confirmed
that, with orbit reduction forced, the checker returns **DB FAIL — a wrong verdict**
(the move actually satisfies detailed balance).

Fix: every next-state position is now checked to be **translation-covariant**
(`is_covariant_pos` — its τ-row coefficient is `+1`, τ-col `+1`, untainted; i.e. it
shifts *with* the lattice). A reflected, absolute, or nonlinear output position is
detected, reported as translational FAIL, and the run falls back to the
all-states BFS (correct for any algorithm). With the guard active, `reflect_move`
correctly returns **DB PASS**. This makes the equivariance assumption behind orbit
reduction a *checked theorem*, not an unstated precondition.

As defence in depth, ordering comparisons on positions (`<`) and equality tests
that mix an absolute position with a covariant one are now **hard errors**, so a
translation cannot silently branch on an absolute coordinate; the only sanctioned
geometric queries remain `pbc_d2` / `same_site` / `pmod`, all τ-checked.

### 5.4 The `Min` positivity assumption (was §2.2)

Unchanged in scope but now explicit: `min_condition` still requires the cleared
`Min[a,b]` difference to be a balanced `(1-exp)` binomial and documents that this
is direction-preserving only where the operands' denominators are positive (true
under VMMC's `eInit<eFwd` guard). A `Min` outside that pattern is a `cant(...)`
hard error rather than a silent mis-orientation, so it cannot cause a wrong
verdict — at worst it refuses to run.

### 5.5 Open chambers only (was §2.4) and supported weight class (was §2.5)

Both are inherent to the (correct) design and unchanged: DB is continuous in the
couplings, so a positive-measure violation always shows up in an open chamber; and
weights outside the `(1-exp)`-denominator rational-function class remain a hard
error, never a guess.

### 5.6 Performance & generality (was §3) and the parallel flag

- Warm compute is now a few seconds (VMMC ≈ 5 s serial, BFS ≈ 1 s on 8 threads).
  The dominant *wall-clock* cost for the small cases is now Julia's per-process
  JIT; the README documents an optional `PackageCompiler` sysimage to remove it.
- **`-parallel`** (with `julia -t auto`) spreads the per-representative BFS and the
  independent per-pair DB check across cores using thread-local interning caches
  and thread-local τ flags. Default is **serial** (no thread-startup cost, runs
  anywhere). A regression test asserts the parallel run yields an identical
  verdict, chamber count, and transition graph to the serial run — parallelism is
  a speed-only change, never a correctness one. (Thread-local interning means
  fewer cross-thread weight merges, which only adds redundant unique weights;
  conditions still deduplicate by value, so the arrangement and verdict are
  identical.)
- Larger systems: added `metropolis_4x4.jl` and `hop_8way_correct.jl` (both 4×4,
  240 states) to exercise scaling and a correct power-of-two pool. Genuinely large
  cases (e.g. 4×4 with four *distinct* species, 43 680 states) remain expensive —
  the approach is exhaustive by design and is a verifier for small representative
  systems, as the original audit noted; `-parallel` and Int128 push the practical
  ceiling up but do not change that asymptotics.

### 5.7 Smaller items (was §4)

- **Determinism** re-confirmed end to end: no RNG; fixed bit-tree BFS; exact
  simplex (deterministic); canonical orderings; the parallel paths combine
  per-thread results order-independently and are tested to match serial.
- **Hash-consing lifetime**: the interning caches are reset per run inside
  `_init_threadlocal!` and kept alive for the run, so `objectid`-based weight
  dedup is valid; this is now stated at the definition.
- **Error messages**: the user-facing `CantHandle` reasons cover the new guards
  (non-covariant move, position comparison, overflow, empty/inverted range,
  maxdepth) with actionable text.
- **Test suite**: `test_db.jl` now has three layers — unit (incl. the exact
  simplex), fail-loud (every "cannot do this exactly" path raises), and all twelve
  examples end-to-end with their known verdicts — so a false PASS *or* a false
  FAIL anywhere breaks the suite (93 assertions).

### 5.8 Residual concerns (honest list)

- **Scaling is still exponential** in particle count / lattice size; this is
  intrinsic to exhaustive verification. Int128 + parallel raise the ceiling, not
  the exponent.
- **τ-soundness still assumes the documented discipline** for the *narrow* escape
  hatch of calling internal helpers (`tau0`, `intval`) directly from a translation;
  the covariance guard and the comparison guards close the practical routes, but a
  user who deliberately extracts a raw integer via an undocumented internal and
  branches on it is outside the contract. This is now the only remaining τ caveat,
  and it does not arise for any algorithm written against the documented API.
- **Int128 overflow** is fail-loud, not auto-promoting; a system large enough to
  overflow aborts cleanly rather than silently widening to BigInt. Re-running such
  a case would need a BigInt build (a one-line change to `Q`).

### 5.9 Graph-verified p4m symmetry reduction of the DB check (NEW optimisation)

Following `ideas.md` §2.3, the detailed-balance check now exploits the lattice's
full symmetry group (translations ⋊ D4) to evaluate the residual on only one pair
per symmetry orbit. The win over the previous design (which already did per-pair
condition projection) is real but bounded: it cuts the VMMC DB check from ≈2.1 s to
≈0.9 s (pairs checked 11088 → 174) and never makes any case slower.

The soundness discipline is the whole point: a symmetry `g` is used **only after
being verified on the COMPUTED transition graph** — `E(g·s)=E(s)` exactly, and every
edge `(s→t)` maps to an edge `(g·s→g·t)` with the identical weight-index multiset
(which, given hash-consing over D4-invariant atoms, is a sufficient exact test for
identical symbolic weights). The algorithm is never trusted; a symmetry that fails
to verify is dropped (more pairs checked, never fewer than correctness needs). This
is exactly why the long-standing "D4 is too slow or needs trust" dilemma is avoided:
the dilemma applies to using D4 to skip the *BFS*, whereas here D4 only prunes
*provably redundant DB-pair evaluations*. The reduction also captures partial
symmetry (column-only translations for a row field; translations-only for a
directional bias). The regression suite asserts, for every example, that the
symmetry-reduced verdict and chamber count equal the all-pairs baseline, and encodes
the D4⟂DB independence facts (an anisotropic defect breaks both; an isotropic defect
breaks DB while keeping full symmetry; a failed symmetry check is never read as a DB
conclusion).

### 5.10 Generalised from D4 to arbitrary p4m subgroups (follow-up)

The initial implementation (§5.9) checked only two generators of D4 (rotate90 and
diagonal reflect), which meant subgroups like D2 (e.g. 180° rotation + axis
reflections) were not discovered unless those generators happened to verify. This
was extended to check all 8 non-identity elements of D4 as individual candidates
(rotate90, rotate180, rotate270, reflect, reflect_h, reflect_v, reflect_ad), so any
subgroup of p4m is now automatically discovered. The cost is 6 extra graph scans per
run, each O(#states + #edges) and dominated by the DB computation itself. The
correctness argument is unchanged: candidates are only admitted if they pass the
exact energy and graph checks. `examples/horizontal_metropolis.jl` is the canonical
test case: column-only Metropolis has D2 symmetry (verified: rotate180, reflect_h,
reflect_v) but not D4 (rotate90 maps column moves to row moves, outside the proposal
set, so it correctly fails verification), yet DB PASS -- confirming that
point-group symmetry and detailed balance are independent in both directions.
