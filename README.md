# DB_Julia — an exact, Julia-native detailed-balance checker for lattice MCMC

DB_Julia proves — exactly, with no sampling and no floating point in the verdict
— whether a lattice Monte-Carlo algorithm satisfies **detailed balance**, the
condition needed to sample the Boltzmann distribution `π(s) ∝ exp(-β E(s))`:

```
T(s→t) · π(s) = T(t→s) · π(t)        for every pair of states (s, t).
```

You hand it a short Julia translation of one MCMC step. It intercepts every
random-number call, reconstructs the **exact symbolic** transition probabilities
by exhaustive enumeration, and checks the detailed-balance equation
algebraically over **every region of coupling-parameter space at once** — for all
values of the couplings, not a sampled few.

It also reports **translational invariance** (via a symbolic lattice offset τ) and
**ergodicity** (graph reachability).

### Trust is the point

A **false PASS is the worst possible outcome**, so the whole pipeline is exact
and *fail-loud*:

- All arithmetic is exact rationals (`Rational{Int128}`), which is **overflow-
  checked** by Julia — an out-of-range computation throws and aborts, it never
  wraps to a wrong number.
- Chamber feasibility is decided by an **exact rational simplex** (no
  floating-point LP, no tolerance). This was the last numerical step in the old
  design; it is now gone.
- The run is **fully deterministic**: no RNG, fixed bit-tree BFS, canonical
  orderings; nothing depends on hash iteration order or thread scheduling.
- Anything the engine cannot represent exactly (an unsupported threshold algebra,
  an incomplete enumeration, a non-integer coordinate, …) is a **hard error**,
  never a guess.

See [AUDIT.md](AUDIT.md) for the full soundness argument and the boundaries.

---

## Quick start

```
julia --project=. check.jl examples/single_metropolis.jl
```

Use multiple cores on a heavy case (optional; default is serial so the tool runs
anywhere with no thread-startup cost):

```
julia --project=. -t auto check.jl examples/vmmc_2d.jl -parallel
```

Run the regression + stress suite (unit pieces, fail-loud paths, and all bundled
examples with their known PASS/FAIL verdicts):

```
julia --project=. test_db.jl
```

---

## Writing an algorithm file

Copy [TEMPLATE.jl](TEMPLATE.jl) and fill in two functions. A file defines exactly
five top-level names:

```julia
const NGRID          = 3
const MAXD2          = 2              # max squared interaction distance kept
const PARTICLE_TYPES = [1, 2, 3]      # the species multiset (repeats = identical)

energy(state::PState)::LinForm = ...            # symbolic energy, linear in couplings
algorithm(rng, state::PState)::PState = ...     # one MCMC step using the primitives
```

A **state** is a `Vector{Particle}`; each `Particle` has coordinates `.r`, `.c`
and a species `.t`. The coordinates are `TauNum` values that secretly carry the
translation offset τ, so translational invariance is measured for free.

**The one rule:** do ordinary arithmetic on coordinates (`p.r + dr`), ask
geometric questions only through `pbc_d2` / `same_site` / `pmod`, and never
compare coordinates with `<` or convert one to a plain `Int` yourself. Those are
the only ways to leak an absolute position into a decision, and the checker
**forbids them** (it errors rather than risk a wrong translational verdict). Build
the next state with plain arithmetic — the harness wraps it onto the torus for you.

| Primitive | Meaning |
|---|---|
| `rand_choice_index!(rng, n)` / `rand_choice!(rng, list)` | uniform choice; exact `1/n` weight |
| `rand_integer!(rng, lo, hi)` | uniform integer in `[lo,hi]`, exact rejection sampling |
| `metropolis!(rng, dE)` | accept w.p. `min(1, exp(-β·dE))`; `dE::LinForm` |
| `accept!(rng, thr)` | accept w.p. a general symbolic threshold `thr::ThExpr` |
| `th_const, th_boltz, th_sub, th_div, th_min, th_piece, c_lt, c_le` | build symbolic thresholds (for cluster algorithms like VMMC) |
| `pbc_d2(p,q,n)` / `same_site(p,q,n)` / `pmod(x,n)` | min-image distance² / occupancy / `Mod` (all τ-checked) |
| `Jc(a,b,d2)` / `Xparam(:fieldH)` | a coupling atom `couplingJ[a,b,d2]` (canonical `a≤b`) / a field parameter |

Translational invariance is **always reported**; it does not by itself fail the
run (an absolute-field algorithm like `quadratic_field` is correctly τ-FAIL yet
DB-PASS). Full lattice symmetry (translations **and** the square point group D4)
is exploited to speed the detailed-balance check, but only after being **verified
on the computed transition graph** — the algorithm itself is never trusted (see
"How it works" §6).

---

## How it works

### 1. State enumeration
All distinct placements of the typed particles on the `n × n` torus are
enumerated and cross-checked against the combinatorial count
`P(S,N) / ∏ multiplicity!`.

### 2. τ-augmented BFS — translational invariance + exact path enumeration
Every coordinate is a `TauNum`: an exact value plus the linear coefficients of two
symbolic offsets `τr, τc`, plus a *taint* flag for any genuinely nonlinear τ term.
A value is translation-invariant ("τ-free") iff its τ-coefficients are zero and it
is untainted. Pairwise differences cancel τ (so distances/occupancy are τ-free);
squaring a coordinate taints it (so an absolute-position field is detected).

The algorithm is replayed against every fixed bit string by a `BitSeqRNG`: each
random primitive consumes bits and multiplies an exact rational path probability,
and each acceptance records its **symbolic threshold** (never a float). The leaves
of this complete decision tree are the `(next_state, weight)` pairs, weights
summing to 1. The real algorithm runs at τ = 0, so substituting τ = 0 in a leaf
gives exactly the real transition — the DB check is therefore correct regardless
of the translational verdict.

Every next-state position is additionally checked to be a genuine
**translation-covariant** lattice position (it shifts *with* the lattice). This is
what makes the orbit-reduction speed-up (below) sound: it is only ever applied to
an algorithm whose transitions are provably translation-equivariant; anything else
(a reflected or absolute move) is flagged and falls back to the all-states path.

### 3. Exact symbolic weights (and the faithful VMMC ratio)
A leaf weight is a rational coefficient times a product of acceptance factors.
VMMC's Whitelam–Geissler frustration test is translated faithfully (two separate
`RandomReal` draws), so a leaf weight can be a genuine *ratio* of exponential sums
— `(1-exp(βΔrev))/(1-exp(βΔfwd))` capped by `Min[·,1]` — that cancels only after
the two factors are multiplied. Weights are carried as exact rational functions

```
Val = numerator / ∏ (1 - exp(-β·Lₖ))
```

so no cancellation is needed mid-computation and none is lost.

### 4. Chambers — the coupling-parameter regions (exact)
The branch conditions (`dE ≤ 0`, the `Min`'s switch, …) are hyperplanes carving
coupling space into open **chambers**. They are enumerated by a BFS over the
arrangement using an **exact rational simplex** for feasibility (the chamber-
adjacency graph is connected, so the BFS is complete given an exact test). Within
each chamber every threshold resolves to a concrete `Val`.

### 5. Detailed balance — exact, with denominators cleared
For each communicating pair `(s,t)` and each chamber the residual

```
Σ_{leaves s→t} weight·exp(-β E_s)  −  Σ_{leaves t→s} weight·exp(-β E_t)
```

is formed, multiplied through by the common `(1-exp)` denominator, and collapsed
to a single Laurent polynomial; DB holds in that chamber iff **every coefficient
is zero** — checked exactly with `Rational{Int128}`. Grouping by exponent vector
handles half-integer exponents (e.g. `exp(-β·dE/2)`) directly.

### 6. Symmetry-reduced DB check — verified on the graph, never assumed
The square lattice's full symmetry group is `p4m` = translations ⋊ D4 (rotations
and reflections). If a lattice symmetry `g` is a symmetry of the *system*, then
`T(g·s → g·t) = T(s→t)` and `π(g·s) = π(s)`, so the detailed-balance residual of
`(g·s, g·t)` is **identical** to that of `(s,t)` — and DB need be checked on only
one pair per symmetry orbit (up to ~`8·n²` fewer pairs).

Crucially, the checker does **not** assume the algorithm has this symmetry; it
**verifies it on the already-computed transition graph**, which is sound and cheap:

- **energy invariance** `E(g·s) = E(s)` (exact integer-vector comparison), and
- **graph equivariance** — every edge `(s→t)` has an edge `(g·s→g·t)` carrying the
  *identical multiset of weight indices*. Because weights are built from
  D4-invariant coupling atoms and hash-consed, equal indices imply *identical*
  symbolic weights (a sufficient, exact test).

Both checks are `O(#states + #edges)` for the generators of `p4m`. A symmetry that
does not verify is simply dropped (more pairs are checked, never fewer than
correctness needs), so this is **speed-only and can never change the verdict** —
proven in the test suite, which checks every example's reduced verdict against the
full all-pairs baseline. The reduction picks up *partial* symmetry too: a
row-dependent field (`quadratic_field`, `broken_field_wrong_accept`) verifies only
column translations, and a directional bias (`broken_biased_direction`) verifies
only translations (its anisotropy correctly fails D4) — yet the DB violation is
still caught in both.

> **D4 and detailed balance are independent.** A D4-asymmetric algorithm can still
> satisfy DB (e.g. horizontal-only Metropolis), and a perfectly D4-symmetric one
> can violate it (e.g. `broken_metropolis_halfbeta`, `broken_variable_pool`). So a
> failed symmetry check is **never** used to conclude anything about DB — it only
> means more pairs are evaluated. The DB verdict is always computed in full.

### Why it is fast
- Exact arithmetic in **`Rational{Int128}`** rather than `BigInt`: the BFS no
  longer allocates a GMP bignum per tiny-integer operation (~3× faster), while
  staying sound (overflow throws).
- **Per-pair condition projection:** a pair is checked once per distinct
  projection of the chambers onto its few active conditions, not once per chamber.
- **Lazy, cached weight evaluation** and **hash-consed thresholds** so the model
  is built from unique weights, deduplicated by object identity.
- **Translation-orbit reduction (BFS):** one representative per translation orbit
  is BFS'd and its leaves are translated to the rest — sound because the covariance
  check guarantees equivariance (falls back to all-states otherwise).
- **Graph-verified `p4m` symmetry reduction (DB check):** detailed balance is
  evaluated on one pair per symmetry orbit, using only symmetries verified on the
  computed graph (§6). For VMMC this cuts the pairs checked from 11088 to 174 and
  the DB-check time from ≈2.1 s to ≈0.9 s.
- **`-parallel`** (with `julia -t auto`) spreads the per-representative BFS and the
  per-pair DB check across cores; the exact LP and the verdict are unchanged.

---

## Examples & timings

All examples live in `examples/`. Verdicts and chamber counts below are the exact
results. The **DB pairs** column shows how many pairs the graph-verified `p4m`
symmetry reduction actually checks versus the total (`sym` = which symmetries
verified: T = translations, D4 = full point group, T_c = column translations only).

| Example | Trans. | Detailed balance | Ergodicity | states | chambers | DB pairs | sym |
|---|---|---|---|---|---|---|---|
| `single_metropolis.jl` | PASS | PASS | PASS | 504 | 48 | 72 / 4536 | T·D4 |
| `kawasaki.jl` | PASS | PASS | FAIL (by design) | 504 | 6 | 18 / 756 | T·D4 |
| `quadratic_field.jl` | **FAIL** (absolute field) | PASS | PASS | 12 | 6 | 8 / 16 | T_c |
| `broken_variable_pool.jl` | PASS | **FAIL** (pool 3 vs 4) | PASS | 72 | 1 | 6 / 252 | T·D4 |
| `broken_8way_hop.jl` | PASS | **FAIL** (pool 7 vs 8) | PASS | 240 | 1 | 20 / 1792 | T·D4 |
| `broken_biased_direction.jl` | PASS | **FAIL** (duplicated dir) | PASS | 504 | 48 | 504 / 4536 | T |
| `broken_metropolis_halfbeta.jl` | PASS | **FAIL** (`β/2`, half-int exp) | PASS | 504 | 48 | 72 / 4536 | T·D4 |
| `broken_field_wrong_accept.jl` | PASS | **FAIL** (accept ignores field) | PASS | 12 | 2 | 8 / 16 | T_c |
| `vmmc_2d.jl` | PASS | PASS | PASS | 504 | 216 | 174 / 11088 | T·D4 |
| `hop_8way_correct.jl` | PASS | PASS | PASS | 240 | 1 | 20 / 1792 | T·D4 |
| `metropolis_4x4.jl` | PASS | PASS | PASS | 240 | 24 | 20 / 1792 | T·D4 |
| `reflect_move.jl` | **FAIL** (non-covariant move) | PASS | FAIL | 72 | 1 | 14 / 42 | T_c |

Three are deliberate edge cases: a **correct** power-of-two pool on a larger 4×4
lattice (`hop_8way_correct`); a larger interacting 4×4 system (`metropolis_4x4`);
and a **non-translation-covariant** move (`reflect_move`) that exercises the
covariance guard — without that guard, orbit reduction would report a *wrong*
verdict on it (see AUDIT.md §5.3). Note how `broken_biased_direction` verifies
only translations (its directional bias breaks D4) and the row-field cases verify
only column translations — the reduction discovers exactly the symmetry the
*graph* actually has.

**Warm compute (JIT excluded), serial** on an Apple-silicon laptop:

| Example | BFS | model | DB check | total |
|---|---|---|---|---|
| `single_metropolis.jl` | 0.19 s | 0.01 s | 1.27 s | ≈ 1.5 s |
| `metropolis_4x4.jl` | 0.04 s | 0.01 s | 0.27 s | ≈ 0.3 s |
| `vmmc_2d.jl` | 3.2 s | 0.05 s | 0.97 s | ≈ 4.2 s |

The graph-verified symmetry reduction makes the VMMC DB check ≈2.1× faster (0.97 s
vs 2.08 s checking all pairs). With `-parallel` on 8 threads the VMMC BFS drops to
≈1 s. A single cold `check.jl` invocation additionally pays **~13–18 s of Julia JIT
compilation** (the engine is recompiled per process); the regression suite
amortises this across all examples, and a `PackageCompiler.jl` system image removes
it entirely (below).

### Removing the JIT warm-up (optional)
Most of the wall-clock for the small cases is Julia compiling the engine afresh
each process. To eliminate it, build a system image once:

```
julia --project=. -e 'using PackageCompiler; create_sysimage(; sysimage_path="dbc.so", \
    precompile_execution_file="check.jl")'
julia --project=. --sysimage dbc.so check.jl examples/vmmc_2d.jl
```

This is optional and machine-specific; the default workflow needs no build step.

---

## Fail-loud guarantees

The checker **aborts with an error** rather than return a possibly-wrong verdict
whenever it meets something it cannot represent exactly:

- a BFS path exceeds `maxdepth` (an incomplete tree would silently drop
  transitions) — raise `maxdepth` with `-maxdepth N`;
- a non-integer or non-covariant position coordinate, or an inverted/empty random
  range;
- a `Min` whose branch is not a genuine linear hyperplane, or a division by a
  non-binomial threshold (outside the supported exact rational-function class);
- an exact-arithmetic overflow (the system is too large for `Int128`);
- any random primitive or comparison that could leak an absolute position.

---

## Files

| File | Purpose |
|---|---|
| `dbc.jl` | the engine (TauNum, BitSeqRNG, exact rational/Val algebra, exact simplex, BFS, DB check) |
| `check.jl` | command-line driver (`[-maxdepth N] [-parallel]`) |
| `test_db.jl` | unit + fail-loud + end-to-end example suite |
| `TEMPLATE.jl` | annotated template for writing your own algorithm |
| `examples/` | twelve worked translations (nine standard + three edge cases) |
| `AUDIT.md` | critical soundness/performance audit and how each issue is addressed |
