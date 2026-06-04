# DB_Julia — a Julia-native detailed-balance checker for lattice MCMC

DB_Julia verifies that a lattice Monte-Carlo algorithm satisfies **detailed
balance** — the condition required to sample the Boltzmann distribution
`π(s) ∝ exp(-β E(s))` correctly:

```
T(s→t) · π(s) = T(t→s) · π(t)    for every pair of states (s, t).
```

It does this **exactly and symbolically**, for all coupling-parameter values at
once, with **no Mathematica and no sampling**. It intercepts every random-number
call the algorithm makes, reconstructs the exact symbolic transition
probabilities by exhaustive enumeration, and checks the detailed-balance
equation algebraically over every region of coupling-parameter space.

It also checks **translational invariance** (algebraically, via a symbolic
lattice offset τ) and **ergodicity** (graph reachability).

The whole pipeline runs in Julia and uses exact `Rational{BigInt}` arithmetic
throughout, so it can never return a false PASS or a false FAIL. Anything it
cannot represent exactly is raised as a hard error rather than guessed.

---

## Quick start

```
julia --project=. check.jl examples/single_metropolis.jl
```

Run the regression suite (proven unit pieces + fast PASS/FAIL end-to-end checks):

```
julia --project=. test_db.jl
```

---

## Writing an algorithm file

An algorithm file is a Julia translation of **one MCMC step**. It defines:

```julia
const NGRID          = 3
const MAXD2          = 2          # max squared interaction distance
const PARTICLE_TYPES = [1, 2, 3]

energy(state::PState)::LinForm = ...            # symbolic energy, linear in couplings
algorithm(rng, state::PState)::PState = ...     # one MCMC step using the rng primitives
```

A **state** is a vector of `Particle(r, c, type)`. The positions `r, c` are
`TauNum` values (lattice coordinates that secretly carry the translation offset
τ — see below). You build moves with the provided random primitives and geometry
helpers; you never touch τ directly.

| Primitive | Meaning |
|---|---|
| `rand_choice!(rng, list)` / `rand_choice_index!(rng, n)` | uniform choice; exact `1/n` rejection-sampling weight |
| `rand_integer!(rng, lo, hi)` | uniform integer in `[lo,hi]`, exact rejection sampling |
| `metropolis!(rng, dE)` | `RandomReal[] < min(1, exp(-β·dE))`; `dE::LinForm` |
| `accept!(rng, thr)` | `RandomReal[] < thr` for a general symbolic threshold `thr::ThExpr` |
| `th_const`, `th_boltz(L)`, `th_sub`, `th_div`, `th_min`, `th_piece`, `c_lt`, `c_le` | build symbolic thresholds (used for cluster algorithms like VMMC) |
| `pbc_d2(p,q,n)` / `same_site(p,q,n)` / `pmod(x,n)` | minimum-image distance² / occupancy / `Mod` (all τ-checked) |
| `Jc(a,b,d2)` / `Xparam(:fieldH)` | a coupling atom `couplingJ[a,b,d2]` (canonicalised `a≤b`) / a field parameter |

Translational invariance is **always checked and reported** — you do not declare
it, and a failure (e.g. an absolute-position field) does not by itself fail the
run. Point-group (D4) symmetry is never used.

---

## How it works

### 1. State enumeration

All distinct placements of the typed particles on the `nGrid × nGrid` torus are
enumerated and cross-checked against the combinatorial count
`P(S,N) / ∏ multiplicity!`.

### 2. τ-augmented BFS — translational invariance + exact path enumeration

Every lattice coordinate is a `TauNum`: an exact value plus the linear
coefficients of two symbolic offsets `τr, τc`, plus a *taint* flag for any
genuinely nonlinear τ term (`τr²`, `τr·τc`, …). A value is **translation
invariant** ("τ-free") iff its τ-coefficients are zero and it is untainted.

- A pairwise **difference** of two positions cancels τ exactly → distances and
  occupancy tests are τ-free.
- **Squaring** a τ-augmented coordinate produces a `τr²` term and taints it →
  an absolute-position field (e.g. `Σ row²`) is detected as *not* translation
  invariant.
- Any `Mod`/comparison/branch on a τ-dependent value is flagged.

The algorithm is replayed against every fixed bit string by a `BitSeqRNG`: each
random primitive consumes bits and multiplies an exact rational path
probability, and each acceptance test reads one bit and records its **symbolic
threshold** (never a float). This enumerates the complete decision tree; the
leaves are the `(next_state, weight)` pairs, with weights summing to 1.

The real algorithm runs at τ = 0, so substituting τ = 0 in a leaf gives exactly
the transition the real lattice algorithm makes — **the detailed-balance check
is therefore correct regardless of the translational verdict** (this is why
`quadratic_field` is τ-FAIL but DB-PASS).

### 3. Exact symbolic weights (and the faithful VMMC ratio)

A leaf weight is a rational coefficient times a product of acceptance factors.
Each factor is a symbolic threshold `ThExpr` (a piecewise function of the
couplings — `Min`, ratios, `exp`, with linear-inequality guards). For a simple
Metropolis move the threshold is `Piecewise[{{1, dE≤0}}, exp(-β·dE)]`.

VMMC's Whitelam–Geissler frustration test is translated **faithfully** (two
separate `RandomReal` draws), so a leaf weight can be a genuine *ratio* of
exponential sums — `(1-exp(βΔrev))/(1-exp(βΔfwd))` capped by `Min[·,1]` — which
only cancels after the two factors are multiplied. The engine carries weights as
exact rational functions

```
Val = numerator / ∏ (1 - exp(-β·Lₖ))
```

(`num` a Laurent polynomial in the exp-monomials, `den` a multiset of
binomials), so no cancellation is needed mid-computation and none is lost.

### 4. Chambers — the coupling-parameter regions

The branch conditions (`dE ≤ 0`, `eInit < eFwd`, the `Min`'s linear switch, …)
are hyperplanes carving coupling space into open **chambers**. They are
enumerated by a BFS over the arrangement using **HiGHS** LP feasibility checks
(non-strict negation bridges all octants; degenerate boundary patterns are
filtered algebraically). Within a chamber every threshold resolves to a concrete
`Val`.

### 5. Detailed balance — exact, with denominators cleared

For each communicating pair `(s,t)` and each chamber, the residual

```
Σ_{leaves s→t} weight·exp(-β E_s)  −  Σ_{leaves t→s} weight·exp(-β E_t)
```

is formed. Multiplying through by the common denominator (a product of
`(1-exp)` binomials, nonzero inside an open chamber) turns it into a single
Laurent polynomial; detailed balance holds in that chamber iff **every
coefficient is zero** — checked exactly with `Rational{BigInt}`. Grouping by
exponent vector is algebraically exact and handles half-integer exponents
(e.g. `exp(-β·dE/2)`) directly.

### Why it is fast (optimisations, profiling-driven)

Four optimisations take VMMC (`nGrid=3`, types `{1,2,3}`) from **355 s to
~26 s** while keeping the result identical (216 chambers, DB PASS):

| Optimisation | What it does | VMMC effect |
|---|---|---|
| **Per-pair condition projection** | a pair is checked once per *distinct projection* of the chambers onto that pair's few active conditions, not once per chamber | DB check 169 s → 6 s |
| **Lazy weight evaluation + unique-weight scan** | leaf `Val`s are computed on demand and cached; the model is built from unique weights, not all leaves | model build 87 s → 4 s |
| **Translation-orbit reduction** | a translation-invariant algorithm has translation-equivariant transitions, so only one representative per orbit is BFS'd and the rest are obtained by translating its leaves (falls back to all-states when τ fails) | BFS over reps, not all states |
| **Hash-consing of thresholds** | structurally-equal thresholds are interned to one object, making weight-deduplication an objectid comparison | weight dedup 81 s → ≈0 |

---

## Fail-loud guarantees

A false PASS is the most dangerous outcome, so the checker **aborts with an
error** rather than return a possibly-wrong verdict whenever it meets something
it cannot represent exactly:

- a BFS path exceeds `maxdepth` (an incomplete tree would silently drop
  transitions);
- a non-integer position coordinate, or an inverted `RandomInteger` range;
- a `Min` whose branch is not a genuine linear hyperplane, or a division by a
  non-binomial threshold (outside the supported exact-rational-function class);
- any random primitive form not listed above.

## Documented edge cases

- **τ-coverage is exact for values built from the typed API.** Positions are
  `TauNum` and must stay `TauNum` until after any τ-sensitive operation; a
  translation that extracts the raw integer early and branches on it would hide
  that τ-dependence. Translations are reviewed, not auto-generated.
- A translation-*covariant* next state (one that shifts *with* the lattice) is
  correctly **not** flagged — only τ-dependence in a weight or branch is a
  violation.
- The supported exact-weight class is rational functions whose denominators are
  products of `(1-exp(-β·L))` binomials (this is exactly what VMMC needs).
  Anything outside it is a hard error.

---

## Examples & profiling

All nine examples in `examples/` (translations of standard lattice algorithms,
including a faithful VMMC). Times are wall-clock for a single `check.jl`
invocation on an Apple-silicon laptop and **include ~4–5 s of Julia JIT warmup
per process** (a `PackageCompiler.jl` system image would remove it).

| Example | Translational | Detailed balance | Ergodicity | states | chambers | time |
|---|---|---|---|---|---|---|
| `single_metropolis.jl` | PASS | PASS | PASS | 504 | 48 | 10.9 s |
| `kawasaki.jl` | PASS | PASS | FAIL (by design) | 504 | 6 | 9.2 s |
| `quadratic_field.jl` | **FAIL** (absolute field) | PASS | PASS | 12 | 6 | 9.2 s |
| `broken_variable_pool.jl` | PASS | **FAIL** (pool 3 vs 4) | PASS | 72 | 1 | 6.6 s |
| `broken_8way_hop.jl` | PASS | **FAIL** (pool 7 vs 8) | PASS | 240 | 1 | 5.7 s |
| `broken_biased_direction.jl` | PASS | **FAIL** (duplicated dir) | PASS | 504 | 48 | 10.5 s |
| `broken_metropolis_halfbeta.jl` | PASS | **FAIL** (`β/2`, half-integer exp) | PASS | 504 | 48 | 10.4 s |
| `broken_field_wrong_accept.jl` | PASS | **FAIL** (accept ignores field) | PASS | 12 | 2 | 8.2 s |
| `vmmc_2d.jl` | PASS | PASS | PASS | 504 | **216** | 26.4 s |

Per-phase breakdown (wall-clock, JIT included):

| Phase | single_metropolis | vmmc_2d |
|---|---|---|
| τ-augmented BFS (reps) | 5.0 s | 16.1 s |
| DB model build | 2.0 s | 4.0 s |
| chambers + DB check | 3.6 s | 6.2 s |

The VMMC chamber count (216) and DB-PASS verdict match the reference
Mathematica implementation exactly. The remaining VMMC cost is the symbolic BFS
of the 56 orbit representatives; the natural next steps are a narrower numeric
type for lattice coordinates (positions are tiny integers, currently carried as
`Rational{BigInt}`) and a precompiled system image to remove JIT warmup.
