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

Check **global balance** (`π·T = π`) instead of the stronger detailed balance — this
accepts correct **non-reversible** samplers that DB rejects (see below):

```
julia --project=. check.jl examples/directed_sweep.jl            # Detailed balance: FAIL
julia --project=. check.jl examples/directed_sweep.jl -balance   # Balance: PASS
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
const MOVES          = [...]          # OPTIONAL: declared displacement set (see below)

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
| `unordered(rng, list)` | iterate `list` order-independently (the OIP contract, below): reads no item content, consumes no bits |
| `metropolis!(rng, dE)` | accept w.p. `min(1, exp(-β·dE))`; `dE::LinForm` |
| `accept!(rng, thr)` | accept w.p. a general symbolic threshold `thr::ThExpr` |
| `th_const, th_boltz, th_linear, th_sub, th_div, th_min, th_max, th_piece, c_lt, c_le` | build symbolic thresholds (for cluster/Barker/rate algorithms) |
| `pbc_d2(p,q,n)` / `same_site(p,q,n)` / `pmod(x,n)` | min-image distance² / occupancy / `Mod` (all τ-checked) |
| `Jc(a,b,d2)` / `Xparam(:fieldH)` | a coupling atom `couplingJ[a,b,d2]` (canonical `a≤b`) / a field parameter |
| `rand_move!(rng)` / `move(p,d)` / `rev(d)` | pick a declared direction / shift covariantly / reverse it (the `MOVES` contract, below) |

**Optional — the `MOVES` contract (point-group reduction).** A move written as
`p.r + dr` with a *hardcoded* `(dr,dc)` is an absolute constant, so the checker
cannot certify rotation/reflection equivariance from it. If instead you **declare**
the displacement set `const MOVES = [...]` and write moves as `move(p, rand_move!(rng))`
(use `rev(d)` for the reverse displacement), directions become covariant objects the
point group permutes — and the checker certifies equivariance under the largest
subgroup of `D4` that `MOVES` is closed under, reducing the τ-BFS over the lattice
point group too (up to 8× on top of translation × species). This is opt-in: omit
`MOVES` and everything behaves as before. See [doc/rotation-taint.md](doc/rotation-taint.md).

**Optional — the `unordered` contract (order-independent iteration).** A move that
processes a *set* of candidates in a loop whose result does not depend on the
visiting order (a cluster builder, say) has two bad options for that loop: a
**sort** that tie-breaks on `state[q].t` (which breaks the species symmetry — this
is exactly why `vmmc_2d` declines species), or an explicit **random shuffle** of the
order (which restores the symmetry but multiplies the decision tree by `|cands|!`
per step). Writing the loop as `for q in unordered(rng, cands)` avoids both: the
primitive yields a canonical order that **reads no item content** (so it can't break
a symmetry) and **consumes no random bits** (so the tree does not blow up). You are
*asserting* order-independence; the checker **verifies** it — it re-BFSes each
representative in **five alternative orders** (the reverse, two cyclic shifts, and
their reverses) and checks the transition probabilities are identical, raising a
**hard error** if your body actually depends on order. So you get the small tree of
the sorted version *and* the full species + point-group symmetry of the shuffled
version. See [examples/vmmc_2d_unordered.jl](examples/vmmc_2d_unordered.jl).

**Soundness scope of the cross-check (important for exotic moves).** The verification
is exact and float-free, and it is **sufficient** because order-dependence in a
cluster move lives in the *diagonal* (self-loop / rejection) leaves, which neither
detailed balance nor global balance uses; only the **off-diagonal** transition leaves
are compared, and these are order-invariant *exactly* for an order-independent move.
Three facts make this robust: (i) **species** relabelling never changes the gather
order (candidates are position-sorted, and a relabel doesn't move positions), so the
species reduction is *unconditionally* sound; (ii) for a candidate list of length
**≤ 3** the five probed orders are **exhaustive** (all permutations), so the check is
*complete* — and on a lattice a cluster particle rarely has more than three
simultaneous in-range neighbours; (iii) the regression suite pins every `unordered`
example against a **direct all-states build** (no derivation), which would catch any
mis-reduction. A move that genuinely depends on order — e.g. terminating cluster
growth *inside* the candidate loop and moving the partial cluster — is correctly
**rejected**; one that stops *after* a whole particle's loop is order-independent and
**accepted** (see [examples/vmmc_early_stop.jl](examples/vmmc_early_stop.jl), which is
order-independent yet **DB- and balance-FAIL** — naive early stopping breaks
sampling, and the checker catches it). Its correct counterpart is
[examples/cluster_metropolis.jl](examples/cluster_metropolis.jl): a cluster move whose
early stopping is **fixed by a final Metropolis acceptance** of the cluster against its
environment (plus the recruitment proposal-ratio) — the original Whitelam–Geissler
idea — which the checker confirms **DB-PASS** (and balance-PASS).

The threshold algebra handles: `exp(-β·linear)` and Laurent polynomials in such
exp-monomials; **`th_linear(L)`** — the bare value `⟨L,J⟩`, so weights may carry
**polynomial-in-coupling factors** (e.g. a rate proportional to a field); division
by any binomial denominator including **`1+exp`** (Barker/Glauber) as well as
`1-exp` (VMMC); and `th_min`/`th_max` whose switch is a single hyperplane. See
[doc/expressiveness.md](doc/expressiveness.md) for the exact supported class.

Translational invariance is **always reported**; it does not by itself fail the
run (an absolute-field algorithm like `quadratic_field` is correctly τ-FAIL yet
DB-PASS). Three symmetries are exploited to speed the check: lattice **translation**,
**species** permutation (relabeling particle types, which permutes the symbolic
couplings), and — when the `MOVES` contract is used — the lattice **point group**
(rotations/reflections, `p4m`). All three reduce the τ-BFS (translation via `τ`,
species via a τ-style label tag, point group via the supplied-direction tag; §2) and
the detailed-balance pair check (§6). All are **verified** (on a single BFS, or on
the computed transition graph), so the algorithm itself is never trusted, and any
subgroup is discovered automatically.

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
what makes the orbit-reduction speed-up sound: it is only ever applied to an
algorithm whose transitions are provably translation-equivariant; anything else (a
reflected or absolute move) is flagged and falls back to the all-states path.

The BFS is further reduced over **species (type-permutation) orbits** when the
algorithm is species-equivariant. During the BFS each species label is wrapped in a
*tag* (the discrete analogue of `τ`): label-equality, hashing and `Jc` are allowed,
but any use of the *absolute* label (compare-to-constant, ordering, arithmetic, or
emitting a fresh-literal output label) is flagged. An unflagged BFS certifies
species-equivariance, so one representative per **combined (translation × species)
orbit** is BFS'd and the rest are derived by translating *and* relabeling — e.g.
56→12 reps. A flag makes the checker fall back (sound; it never assumes).

When the algorithm declares its displacement set (`const MOVES`, the contract above)
the BFS is reduced over the lattice **point group** as well. Directions drawn via
`rand_move!` and applied with `move`/`rev` are tagged the same way labels are: any
*absolute* use of a direction or a bare position offset is flagged, and the static
subgroup of `D4` under which `MOVES` is closed is computed without running the
rotated states. An unflagged BFS then certifies point-group equivariance, so one
representative per **combined (translation × species × point-group) orbit** is BFS'd
and the rest are derived by translating, relabeling *and* rotating (rotation
preserves distances and types, so it leaves the symbolic weights unchanged) — e.g.
single-Metropolis 56→**4**, VMMC-shuffle 12→**4**. Falls back per element of `D4` if
flagged; the discovered subgroup is exactly right (e.g. column-only moves → D2). The
derivation is validated to reproduce a direct all-states build exactly.

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
Two kinds of symmetry are exploited, both **verified on the already-computed
transition graph** (never assumed of the algorithm), so they are speed-only and
can never change the verdict:

**Spatial (`p4m` = translations ⋊ D4).** If a lattice symmetry `g` is a symmetry of
the *system* then `T(g·s→g·t) = T(s→t)` and `π(g·s) = π(s)`, so the DB residual of
`(g·s, g·t)` is **identical** to that of `(s,t)`. Verified by: **energy invariance**
`E(g·s) = E(s)` (exact integer-vector comparison) and **graph equivariance** — every
edge `(s→t)` has an edge `(g·s→g·t)` with the *identical multiset of weight indices*
(weights are hash-consed over distance-based atoms, so equal indices ⇒ identical
symbolic weights). All 8 D4 elements are tried individually, so any subgroup (D4,
D2, C4, …) is found.

**Species (type permutation).** A permutation `σ` of the species labels relabels the
symbolic coupling atoms — a *bijection of coupling space* — so it too is a symmetry
of the DB problem when the algorithm is species-equivariant: then
`R_{σs,σt}(J) = R_{s,t}(σ⁻¹·J)`, and a residual that is identically zero stays so
under a coordinate bijection. The same graph verification is used, now **up to the
atom relabeling**: energy must equal the σ-permuted energy, and each edge's weights
must equal the σ-*relabeled* originals (each weight's atoms are permuted, re-interned,
and looked up — index 0 if absent ⇒ candidate rejected). Generators are the
multiplicity-preserving label transpositions, so `[1,2,3]→S₃`, `[1,2]→S₂`, and
unequal multiplicities give nothing. `vmmc_2d` is correctly declined (its cluster
sort tie-breaks on the type label).

A candidate that does not verify is dropped (more pairs checked, never fewer). The
full reduction uses one pair per orbit of `(verified spatial) × (verified species)`.
The test suite checks every example's reduced verdict against the all-pairs baseline.

> **Point-group symmetry and detailed balance are independent.** An algorithm with
> only D2 symmetry (e.g. `horizontal_metropolis`, column-moves only) can still
> satisfy DB. A perfectly D4-symmetric algorithm can violate DB (e.g.
> `broken_metropolis_halfbeta`, `broken_variable_pool`). A failed symmetry check is
> **never** used to conclude anything about DB — it only means more pairs are
> evaluated. The DB verdict is always computed in full.

### 7. Global balance (`-balance`) — the weaker, *necessary* condition (exact)
Correct sampling requires only **stationarity** `π·T = π` (global balance); detailed
balance is a stronger, *sufficient* condition. An entire family of modern samplers —
event-chain, lifting, Suwa–Todo — deliberately **violates DB** to mix faster, yet
samples `π` correctly. Passing `-balance` checks balance instead of DB, so those are
accepted.

Balance is the **column sum** of the DB residual matrix: for each target state `t`,
```
B_t  =  Σ_s [ T(s→t)·π(s) − T(t→s)·π(t) ]  =  0 .
```
The `s = t` term cancels, so only off-diagonal transitions enter (exactly what the
graph stores). The check reuses the **same exact rational machinery** as DB — it sums
the directed leaf contributions over a *common* `(1−exp)` denominator and tests that
the numerator's coefficients all vanish in `ℚ` — so it is **exact and float-free**,
just like the DB check. It is symmetry-reduced the same way (one target per verified
state-orbit, since `B_{g·t} ≡ 0 ⇔ B_t ≡ 0`). `examples/directed_sweep.jl` is the
canonical demonstration: a directed shift is **DB-FAIL** but **balance-PASS** (a cyclic
permutation of states keeps the uniform `π` stationary). DB still implies balance, so
every DB-PASS example also passes `-balance`. Default behaviour is unchanged (detailed
balance); `-balance` is opt-in.

### Why it is fast
- Exact arithmetic in **`Rational{Int128}`** rather than `BigInt`: the BFS no
  longer allocates a GMP bignum per tiny-integer operation (~3× faster), while
  staying sound (overflow throws).
- **Per-pair condition projection:** a pair is checked once per distinct
  projection of the chambers onto its few active conditions, not once per chamber.
- **Lazy, cached weight evaluation** and **hash-consed thresholds** so the model
  is built from unique weights, deduplicated by object identity.
- **Translation + species + point-group orbit reduction (BFS — the bottleneck):** one
  representative per *combined* (translation × species × point-group) orbit is BFS'd and
  its leaves are translated, species-relabeled *and* rotated to the rest.
  Translation-equivariance is certified by `τ` + the position-covariance guard;
  species-equivariance by a `τ`-style **species tag** + a species-covariance guard;
  point-group equivariance by a **direction tag** (the `MOVES` contract) + static
  `D4`-closure — all during that single BFS. Falls back per symmetry (then all-states)
  if a guard fires. E.g. randomised-order VMMC drops 56→**4** reps (BFS ≈2.5 s → ≈0.8 s),
  single-Metropolis 56→**4**; declining algorithms (sorted VMMC declines *species* but
  still gets D4: 56→8) are handled soundly.
- **Graph-verified symmetry reduction (DB check):** detailed balance is evaluated
  on one pair per orbit of the verified `(spatial p4m) × (species permutation)`
  group (§6) — any subgroup is discovered automatically from the computed graph.
  For VMMC this cuts pairs from 11088 to 174 (DB-check ≈2.1 s → ≈0.9 s); adding
  species symmetry cuts single-Metropolis 72→16 and kawasaki 18→6. (The pair-count
  drop is large, but wall-time gain is bounded — the DB cost is dominated by the
  distinct-weight evaluations, not the pair count.)
- **`-parallel`** (with `julia -t auto`) spreads the per-representative BFS and the
  per-pair DB check across cores; the exact LP and the verdict are unchanged.

---

## Examples & timings

All examples live in `examples/`. Verdicts and chamber counts below are the exact
results. **BFS'd** = states actually enumerated by the τ-BFS vs the total (the
translation × species × point-group orbit reduction); **DB pairs** = pairs the
graph-verified symmetry reduction checks vs the total. The `sym` column lists the
verified **spatial** group (T = translations, D4 = full point group, T_c = column-only,
D2 = `{rot180, reflect_h, reflect_v}`) and the verified **species** group acting on
the type labels (`S₃`, `S₂`, …). A ★ marks an example that uses the `MOVES` contract,
so its BFS is reduced over the point group too.

| Example | Trans. | Detailed balance | Ergodicity | states | BFS'd | DB pairs | sym (spatial × species) |
|---|---|---|---|---|---|---|---|
| `single_metropolis.jl` ★ | PASS | PASS | PASS | 504 | **4** | 16 / 4536 | p4m × S₃ |
| `kawasaki.jl` ★ | PASS | PASS | FAIL (by design) | 504 | **4** | 6 / 756 | p4m × S₃ |
| `quadratic_field.jl` | **FAIL** (absolute field) | PASS | PASS | 12 | 12 | 4 / 16 | T_c, reflect_v × S₂ |
| `broken_variable_pool.jl` | PASS | **FAIL** (pool 3 vs 4) | PASS | 72 | 4 | 3 / 252 | p4m × S₂ |
| `broken_8way_hop.jl` | PASS | **FAIL** (pool 7 vs 8) | PASS | 240 | 9 | 10 / 1792 | p4m × S₂ |
| `broken_biased_direction.jl` | PASS | **FAIL** (duplicated dir) | PASS | 504 | 12 | 46 / 4536 | T, reflect_h × S₃ |
| `broken_metropolis_halfbeta.jl` | PASS | **FAIL** (`β/2`, half-int exp) | PASS | 504 | 12 | 16 / 4536 | p4m × S₃ |
| `broken_field_wrong_accept.jl` | PASS | **FAIL** (accept ignores field) | PASS | 12 | 3 | 4 / 16 | T_c, reflect_v × S₂ |
| `vmmc_2d.jl` ★ | PASS | PASS | PASS | 504 | **8** | 174 / 11088 | p4m (species declined) |
| `hop_8way_correct.jl` ★ | PASS | PASS | PASS | 240 | **5** | 10 / 1792 | p4m × S₂ |
| `metropolis_4x4.jl` ★ | PASS | PASS | PASS | 240 | **5** | 10 / 1792 | p4m × S₂ |
| `reflect_move.jl` | **FAIL** (non-covariant move) | PASS | FAIL | 72 | 72 | 4 / 42 | T_c, reflect_v × S₂ |
| `horizontal_metropolis.jl` ★ | PASS | PASS | FAIL (rows fixed) | 72 | **3** | 3 / 126 | **D2** × S₂ |
| `barker_accept.jl` | PASS | PASS | PASS | 504 | 12 | 16 / 4536 | p4m × S₃ |
| `poly_rate_accept.jl` | PASS | PASS | PASS | 504 | 12 | 16 / 4536 | p4m × S₃ |
| `vmmc_2d_shuffle.jl` ★ | PASS | PASS | PASS | 504 | **4** | 44 / 11088 | p4m × S₃ |
| `vmmc_2d_unordered.jl` ★ | PASS | PASS | PASS | 504 | **4** | 174 / 11088 | p4m × S₃ |
| `hop_repeated_species.jl` | PASS | PASS | PASS | 756 | 42 | … | p4m × **S₂ (1↔2)** |
| `broken_species_halfbeta.jl` | PASS | **FAIL** (species-dep. `β/2`) | PASS | 504 | 56 | … | T·D4 (species declined) |
| `swap_literal_species.jl` | PASS | PASS | FAIL (N! perms) | 504 | 56 | … | T·D4 (species declined) |
| `directed_sweep.jl` | PASS | **FAIL** (non-reversible) | FAIL (directed) | 9 | 1 | — | T, reflect_h (**balance PASS** with `-balance`) |
| `vmmc_early_stop.jl` ★ | PASS | **FAIL** (early stop) | PASS | 504 | **4** | … | p4m × S₃ (OIP-accepted, order-indep.; **balance FAIL** too) |
| `cluster_metropolis.jl` ★ | PASS | PASS | PASS | 504 | **4** | … | p4m × S₃ (early stop **fixed** by a final Metropolis vs environment) |

`vmmc_2d_shuffle` is the headline: with the `MOVES` contract its BFS drops 56→**4**
reps (warm BFS ≈2.5 s → ≈0.8 s) — species *and* full-D4 reduction. `vmmc_2d` declines
*species* (its sort tie-breaks on the type label) but still gets the full point group
(56→8) — a clean demonstration that the two reductions are independent.
`horizontal_metropolis` (column-only moves) discovers exactly **D2**, not D4. The
last few rows are particle-swap / species edge cases. The `sym` column shows exactly
what the *graph* has, not what was assumed.
`broken_biased_direction` gets `reflect_h` but not `reflect_v`/`rotate90` (the
column bias survives a row-flip, not a column-flip). The row-field examples
(`quadratic_field`, `broken_field_wrong_accept`, `reflect_move`) get column
translation + `reflect_v` only. `horizontal_metropolis` has **D2 not D4**.

**Species (type) symmetry — reduces BOTH the BFS and the DB check.** A permutation
of the species labels relabels the symbolic coupling atoms, so it is a symmetry of
the DB problem when the algorithm is species-equivariant. It is used in two places,
both sound (never trusting the algorithm):

- **τ-BFS reduction (the bottleneck).** During the BFS each label is wrapped in a
  *tag* (the species analogue of `τ`): equality between labels, hashing and `Jc`
  atom-construction are allowed, but any use of the *absolute* label (compare to a
  constant, order, arithmetic) is flagged, and every output label must be an
  inherited tag (species-covariance, the analogue of the position-covariance guard).
  If a representative's BFS runs unflagged it is certified species-equivariant, so
  the BFS runs one rep per **combined (translation × species) orbit** and derives
  the rest by translating *and* relabeling — e.g. 56→12 reps for the S₃ cases.
- **DB-pair reduction.** The same symmetry, re-verified on the *computed graph* up
  to the atom relabeling (§6), checks one pair per orbit.

All-distinct triples (`[1,2,3]`) give `S₃`; `[1,2]` and equal-multiplicity pairs
like `[1,1,2,2]` give `S₂`; unequal multiplicities give none. `vmmc_2d` declines
(its cluster sort tie-breaks on the type label, so it is not *leaf-level*
species-equivariant — sound, never assumed); `vmmc_2d_shuffle` (random order)
does not, and gets the full reduction.

Edge-case examples: a correct power-of-two pool on 4×4 (`hop_8way_correct`); a
larger 4×4 system (`metropolis_4x4`); a non-translation-covariant move
(`reflect_move`, AUDIT §5.3); **`horizontal_metropolis`** — point-group symmetry
and DB are independent (D2 not D4, yet DB-PASS); and four **particle-swap / species**
cases: `vmmc_2d_shuffle` (species-equivariant VMMC), `hop_repeated_species`
(`[1,1,2,2]`, repeated-multiplicity S₂), `broken_species_halfbeta` (species-dependent
acceptance → species declined, DB-FAIL still caught), and `swap_literal_species`
(a swap that writes raw labels → the species-covariance guard declines, DB still
correct). The species reduction is validated to produce a transition graph
*identical* to a direct build (the suite compares them at random coupling points).

`barker_accept` and `poly_rate_accept` exercise the extended weight algebra: Barker
acceptance uses a `1+exp` denominator (no conditions, so a single chamber), and the
rate-limited Metropolis carries a polynomial coupling factor (`a`, the 7th atom)
times the Boltzmann exponentials.

**Warm compute (JIT excluded), serial** on an Apple-silicon laptop. The BFS phase
benefits from the species-orbit reduction where it applies (fewer reps); the `model`
phase runs the species verification (a quick bail when declined).

| Example | BFS | model | DB check | total | BFS'd |
|---|---|---|---|---|---|
| `single_metropolis.jl` | 0.03 s | 0.02 s | 1.0 s | ≈ 1.1 s | 4/504 |
| `kawasaki.jl` | 0.01 s | 0.01 s | ~0 s | ≈ 0.04 s | 4/504 |
| `metropolis_4x4.jl` | 0.02 s | 0.01 s | 0.29 s | ≈ 0.3 s | 5/240 |
| `hop_repeated_species.jl` | 0.11 s | 0.01 s | ~0 s | ≈ 0.12 s | 42/756 |
| `barker_accept.jl` | 0.08 s | 0.02 s | ~0 s | ≈ 0.1 s | 12/504 |
| `poly_rate_accept.jl` | 0.10 s | 0.03 s | 1.2 s | ≈ 1.3 s | 12/504 |
| `vmmc_2d.jl` | 0.61 s | 0.06 s | 0.89 s | ≈ 1.6 s | 8/504 |
| `vmmc_2d_shuffle.jl` | 0.77 s | 0.18 s | 0.92 s | ≈ 1.9 s | 4/504 |
| `vmmc_2d_unordered.jl` | 0.75 s | 0.06 s | 0.80 s | ≈ 1.6 s | 4/504 |

The **species τ-BFS reduction** is the headline: a species-equivariant algorithm
BFSes one rep per combined orbit, e.g. `vmmc_2d_shuffle` (a *typical* randomised-order
VMMC) drops from ≈6.3 s to ≈2.5 s and `single_metropolis` BFS from 0.18 s to 0.08 s;
`vmmc_2d` (declines species) is unchanged. The polynomial-coefficient ring and
`1+exp` denominators add no measurable cost (constant coefficients take a fast path).
With `-parallel` on 8 threads the VMMC BFS drops further. A single cold `check.jl`
invocation additionally pays **~13–18 s of Julia JIT compilation** (the engine is
recompiled per process); the regression suite amortises this, and a
`PackageCompiler.jl` system image removes it entirely (below).

**The `unordered` primitive (OIP) — small tree *and* full symmetry.** `vmmc_2d`
declines species because its candidate sort tie-breaks on the type label;
`vmmc_2d_shuffle` restores species (and gets D4) by drawing a random visiting order,
but that shuffle multiplies the decision tree by `|cands|!` per cluster step.
`vmmc_2d_unordered` uses `for q in unordered(rng, cands)` to get **both** — the
species + point-group symmetry of the shuffle with **none** of its tree blow-up
(`unordered` consumes no bits). On the bundled 3×3 system the candidate sets are tiny
(`|cands| ≤ 2`), so the three VMMC variants are within ~15 % of each other (above);
the OIP win is **asymptotic in the candidate-set size**. Measured directly on a
single cluster step whose seed has `k` mutually-candidate spectators (shuffle vs
`unordered`, leaves enumerated by the τ-BFS):

| `k` (candidates) | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| shuffle leaves (`k!·2ᵏ`) | 8 | 48 | 384 | 3 840 | 46 080 | 645 120 |
| `unordered` leaves (`2ᵏ`) | 4 | 8 | 16 | 32 | 64 | 128 |
| **reduction** (`= k!`) | 2× | 6× | 24× | 120× | 720× | **5040×** |

so a dense cluster move that a shuffle makes intractable stays flat under `unordered`.
The cost is a per-run **cross-check**: when `unordered` is used, the engine re-BFSes
each representative in a second candidate order and asserts the transition
probabilities are identical (a hard error otherwise), so a body that is *not* actually
order-independent is caught rather than reduced unsoundly.

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
| `check.jl` | command-line driver (`[-maxdepth N] [-parallel] [-balance]`) |
| `test_db.jl` | unit + fail-loud + end-to-end example suite |
| `TEMPLATE.jl` | annotated template for writing your own algorithm |
| `examples/` | twenty-three worked translations (standard algorithms + symmetry / weight-class / particle-swap / order-independence / early-termination / non-reversible edge cases) |
| `AUDIT.md` | critical soundness/performance audit and how each issue is addressed |
| `doc/expressiveness.md` | the exact class of weight/energy/condition functions handled |
| `ideas.md` | analysis of inductive generalisation across system sizes, and D4 |
| `type-taint.md` + `doc/typetaint_poc.jl` | the species-equivariance certificate for the BFS (PoC + the analysis that led to it; now implemented) |
| `doc/rotation-taint.md` + `doc/rotation_taint*_poc.jl` | the point-group (D4) BFS reduction via the supplied-direction `MOVES` contract (PoCs + analysis; now implemented) |
| `doc/dbc_method.{tex,pdf}` | a 2-page method writeup |
