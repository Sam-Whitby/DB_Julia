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
DB-PASS). Point-group (D4) symmetry is never used.

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

### Why it is fast
- Exact arithmetic in **`Rational{Int128}`** rather than `BigInt`: the BFS no
  longer allocates a GMP bignum per tiny-integer operation (~3× faster), while
  staying sound (overflow throws).
- **Per-pair condition projection:** a pair is checked once per distinct
  projection of the chambers onto its few active conditions, not once per chamber.
- **Lazy, cached weight evaluation** and **hash-consed thresholds** so the model
  is built from unique weights, deduplicated by object identity.
- **Translation-orbit reduction:** one representative per orbit is BFS'd and its
  leaves are translated to the rest — sound because the covariance check
  guarantees equivariance (falls back to all-states otherwise).
- **`-parallel`** (with `julia -t auto`) spreads the per-representative BFS and the
  per-pair DB check across cores; the exact LP and the verdict are unchanged.

---

## Examples & timings

All examples live in `examples/`. Verdicts and chamber counts below are the exact
results; wall-clock is a single `check.jl` invocation on an Apple-silicon laptop
and **includes ~6–8 s of Julia JIT warm-up per process** (a `PackageCompiler.jl`
system image removes it — see below).

| Example | Trans. | Detailed balance | Ergodicity | states | chambers |
|---|---|---|---|---|---|
| `single_metropolis.jl` | PASS | PASS | PASS | 504 | 48 |
| `kawasaki.jl` | PASS | PASS | FAIL (by design) | 504 | 6 |
| `quadratic_field.jl` | **FAIL** (absolute field) | PASS | PASS | 12 | 6 |
| `broken_variable_pool.jl` | PASS | **FAIL** (pool 3 vs 4) | PASS | 72 | 1 |
| `broken_8way_hop.jl` | PASS | **FAIL** (pool 7 vs 8) | PASS | 240 | 1 |
| `broken_biased_direction.jl` | PASS | **FAIL** (duplicated dir) | PASS | 504 | 48 |
| `broken_metropolis_halfbeta.jl` | PASS | **FAIL** (`β/2`, half-int exp) | PASS | 504 | 48 |
| `broken_field_wrong_accept.jl` | PASS | **FAIL** (accept ignores field) | PASS | 12 | 2 |
| `vmmc_2d.jl` | PASS | PASS | PASS | 504 | 216 |
| `hop_8way_correct.jl` | PASS | PASS | PASS | 240 | 1 |
| `metropolis_4x4.jl` | PASS | PASS | PASS | 240 | 24 |
| `reflect_move.jl` | **FAIL** (non-covariant move) | PASS | FAIL | 72 | 1 |

The last three are added edge cases: a **correct** power-of-two pool on a larger
4×4 lattice (`hop_8way_correct`); a larger interacting 4×4 system
(`metropolis_4x4`); and a **non-translation-covariant** move (`reflect_move`) that
exercises the covariance guard — without that guard, orbit reduction would report
a *wrong* verdict on it (see AUDIT.md §New).

Warm (JIT excluded) compute, serial: VMMC ≈ 5 s total (BFS ≈ 3 s, DB ≈ 2 s);
single-Metropolis ≈ 5 s. With `-parallel` on 8 threads the VMMC BFS drops to ~1 s.

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
