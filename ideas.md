# ideas.md — Can the verifier generalise? And can D4 be exploited?

A brutally honest exploration of two questions about DB_Julia:

1. Given a **PASS** on a small system (say 3×3, 4 particles), can we *prove* the
   algorithm satisfies detailed balance (DB) on **any** lattice size and/or **any**
   particle number — by inductive/structural reasoning over the BFS tree, the way
   τ already rules translational invariance in or out?
2. Is **D4** (square point-group) symmetry exploitable for speed the way
   translational symmetry is — soundly, without trusting the algorithm?

The short answers: **(1)** Naive induction is *false* in general, but there is a
genuine, checkable conditional theorem for *local* updates; the checker today
extracts only one of its three hypotheses (τ). **(2)** D4 cannot soundly accelerate
the *BFS* without either trusting the algorithm or paying the full cost — which is
exactly the historical experience — but it *can* soundly accelerate the *DB-check
phase* by verifying equivariance of the **computed graph** (not the code). That
angle appears under-explored and is the only honest win on the table.

---

## Part 1 — Generalisation by induction

### 1.1 What the checker actually proves, and the two kinds of "for all"

For fixed geometry `(n, N)` the checker proves a statement that is **universal in
the couplings** `J` and **exhaustive in the states**:

> for every communicating pair `(s,t)` and every `J ∈ ℝ^{|A|}`,
> `T(s→t) e^{-βE(s)} = T(t→s) e^{-βE(t)}`.

It is *not* universal in geometry: it says nothing, by itself, about `(n+1, N)` or
`(n, N+1)`. The whole question is whether geometry can be lifted from *instance* to
*universal* the way couplings already are.

The reason couplings lift for free is that `E` is **linear** in `J` and the weights
are exact rational functions of `e^{-βL·J}`; a finite chamber decomposition makes
"all `J`" a finite algebraic check. Geometry has no such finite handle a priori —
that is the crux.

### 1.2 The naive argument, and why it is wrong

The tempting argument is: *the RNG is enumerated exhaustively and deterministically,
so "every route through the code" is taken; a larger lattice cannot create a new
route, so a PASS must persist.*

This conflates **control-flow coverage** with **context coverage**. Even if every
*branch of the code* is executed on 3×3, a larger lattice realises **new geometric
contexts** that change the *data* flowing through those same branches:

- **New interaction distances.** On an `n×n` torus the achievable minimum-image
  squared distances are limited. On 3×3, offsets are `{-1,0,1}`, so `d² ∈ {1,2}`
  *only*; the coupling atom `Jc(a,b,4)` (offset `(2,0)`) **never occurs**. On 5×5 it
  does. An algorithm that mishandles the `d²=4` interaction (wrong sign, wrong
  exponent) **passes on 3×3 and fails on 5×5** — a clean, concrete counterexample.
  *(This is not hypothetical: our own `metropolis_4x4.jl` uses `MAXD2=4` and its
  `d²=4` atom is simply absent from any 3×3 run.)*
- **New occupancy / pool contexts.** Variable-pool bugs (`broken_variable_pool`,
  `broken_8way_hop`) only manifest when both pool sizes occur. A small *dense*
  torus may realise only some pool sizes; `broken_8way_hop` in fact *needs* `n≥4`
  (its README note) precisely because non-8-adjacent pairs do not exist on 3×3.
- **New multi-particle arrangements.** On 3×3 with 4 particles, every site's
  8-neighbourhood is the entire rest of the lattice, so **every pair always
  interacts** — the "two particles far apart, non-interacting" context, which
  dominates large lattices, *cannot be realised at all*. Adding a particle creates
  arrangements (a particle with 3+ simultaneous neighbours) that no smaller `N`
  produces.

So the small lattice generally realises a **strict subset** of the contexts of a
larger one. "All routes covered" is false at the level that matters.

### 1.3 What *does* generalise: locality + saturation

There is nonetheless a real theorem, and it is the right way to think about this.
The key structural fact is **locality of the DB equation for local moves**.

For a **single-particle move** `s→t` (one particle hops from site `x` to `y`):

- The proposal probability is a combinatorial constant `1/(N·D)` times an
  occupancy-dependent factor, where `D=|disps|`. The factor `1/N` and the global
  part of the energy **cancel** in the DB ratio.
- The acceptance depends only on `ΔE = E(t)-E(s)`, which by the pairwise form
  involves **only pairs containing the moved particle within range** — i.e. only
  the other particles inside the radius-`r` neighbourhood of `x` and `y`
  (`r = ⌈√MAXD2⌉`). At most `(#sites within r)` of them, **independent of `N` and
  `n`**.

Therefore the DB equation for a single-particle move is a **local identity**: it is
a function of the *finite local context* (the relative offsets and types of the
≤`k_r` particles within radius `r`, where `k_r` depends only on `r`). There are
**finitely many** such local contexts. If every one of them is realised on the
test system, the per-context identity is verified once and **holds on every `(n,N)`**.

This gives a genuine conditional generalisation theorem:

> **Theorem (informal).** Let an update be (H1) **translation-invariant**, (H2)
> **`r`-local** — every leaf weight and successor of a move depends only on the
> moved particle(s) and other particles within radius `r`, via translation-
> invariant rules — and (H3) such that the test geometry `(n₀,N₀)` is
> **`r`-saturated**: it realises every local context (offset/type multiset within
> `r`, with unambiguous minimum images) that any `(n,N)` with `n≥n₀`, `N≥N₀`
> realises. Then **DB on `(n₀,N₀)` ⟹ DB on all such `(n,N)`**.

*Sketch.* By (H1)+(H2) each move's DB residual is a function only of its local
context; by (H3) every context on the large system equals one on `(n₀,N₀)`, where
the residual was verified zero. The global graph DB is the conjunction of these
local identities. ∎

Two remarks sharpen this:

- **For Metropolis there are two routes, and they need different hypotheses.**
  *(A, complete but demanding)* realise **every local arrangement** (H3): then every
  `ΔE` form — including new linear combinations from more simultaneous neighbours,
  e.g. `2J₁+J₂` — is produced and its residual verified directly. A dense 3×3 with
  3 particles does **not** do this (a move there sees ≤2 neighbours). *(B, weaker
  hypotheses)* exploit that with correct Metropolis acceptance the residual is the
  *algebraic* identity `min(1,e^{-βΔE})e^{-βE_s}=min(1,e^{βΔE})e^{-βE_t}`, true for
  *any* `ΔE`. Then one needs only **atom-saturation** (which atoms exist) plus the
  *structural* fact that the acceptance exponent **is** the declared `ΔE` as a
  function — which the checker can see symbolically (the threshold is literally
  built from `energy(new)-energy(old)`) and, with τ + locality, could lift to a
  universal. Per-instance PASS alone gives this only on the realised forms; turning
  it into a universal is exactly the certificate work of §1.6. Route (B) is why
  single-particle Metropolis is so much more generalisable than its arrangement
  coverage would suggest.
- **Saturation is geometric and checkable.** It needs (a) `n₀ ≥ 2⌈√MAXD2⌉+1` so
  minimum images are unambiguous and there is no spurious wraparound (and no `L/2`
  double-image tie — note `n=4` with `MAXD2=4` is a *special* lattice for exactly
  this reason); (b) enough sites/particles to place every local arrangement. The
  checker can compute the realised atom set and compare it to the infinite-lattice
  set within `MAXD2`; it can also enumerate the local contexts it actually saw.

### 1.4 Where it breaks: unbounded locality (VMMC) and `N`

The theorem needs a **finite locality radius `r`**. This holds for single-particle
Metropolis, Kawasaki (radius around the two swapped sites), and any bounded-range
local move. It **fails for VMMC in `N`**: a virtual-move cluster can chain through
arbitrarily many particles, so its effective radius grows with `N`. A cluster of 5
particles is a genuinely new object that no 4-particle run contains, and the global
move weight is a product over a link structure of unbounded size.

This does **not** mean VMMC is wrong for larger `N` — Whitelam–Geissler VMMC
satisfies DB by construction, and the construction is *pairwise* (every link uses a
pairwise `eInit/eFwd/eRev` and a pairwise frustration test). One can argue that if
**every pairwise link context** is verified and the cluster weight telescopes
(forward links ↔ reverse links), DB lifts to all `N`. But that telescoping is a
*proof about the construction*, not something the current per-instance checker
establishes; certifying it automatically would require recognising the product
structure of the cluster weight and a reverse-move bijection — well beyond τ-style
tagging. So for VMMC the honest statement is: **`N`-generalisation is a theorem
about the algorithm, not a consequence the checker can currently deliver.**

`N`-generalisation for *bounded-range* moves is more hopeful: the relevant-particle
count per move is capped by sites-in-range, not `N`, so a sufficiently dense test
system can saturate it. But "sufficiently dense" fights "unambiguous minimum
image" (dense small tori wrap), so the smallest saturating system is often *larger*
than the smallest interesting one — you may need, say, 5×5 with several particles,
not 3×3 with 4.

### 1.5 Halting / Rice, honestly

The general question — "does this arbitrary algorithm satisfy DB for *all* `(n,N)`?"
— is **undecidable**. The update is effectively arbitrary code; "holds for all
input sizes" is a non-trivial semantic property, so Rice's theorem applies, and
one can encode unbounded search into an "algorithm" whose DB-for-all-`n` is
equivalent to a halting question. So **no procedure can certify universal
generalisation for arbitrary updates.**

This is *not* a counsel of despair. It means generalisation must be sought on a
**restricted, syntactically-recognisable class** — exactly what DB_Julia's API
already imposes (positions are `TauNum`, geometry only via `pbc_d2`/`same_site`,
randomness only via the listed primitives). On that class, locality and saturation
are *analyzable*, and the conditional theorem of §1.3 is the right target. The
undecidability lives in the gap between "arbitrary code" and "the checkable class";
the engineering question is how much of the class we can certify.

### 1.6 What the checker could extract (a concrete, τ-analogous proposal)

τ works because translation acts by a **linear tag that propagates through one BFS
and cancels in differences**, exposing (non-)invariance without re-running. Two
further structural facts admit similar, *single-BFS* certificates:

- **Locality radius `r` — by particle tagging.** Give each particle a formal
  identity tag (not just its position). During the BFS, record, for each leaf
  weight (its conditions and exponents) and each successor, **which particles'
  data it depends on**. If, across all moves, every weight depends only on the
  moved particle(s) and others within radius `r`, the move is certified `r`-local.
  This is the direct analogue of τ-tracking applied to "data support" rather than
  "translation offset", and it is mechanical (the dependency set is already
  implicitly there in the coupling atoms and conditions).
- **Atom/context saturation — by geometry.** Compute the realised atom set and the
  realised local contexts; compare against the closed-form infinite-lattice sets
  within `MAXD2` and against the count of arrangements of `k_r` particles in a
  radius-`r` ball. Emit a verdict: *"saturated at radius `r`; PASS generalises to
  all `n ≥ n₀`, `N ≥ N₀`"* or *"not saturated — atom `Jc(a,b,4)` / a pool size /
  an arrangement is untested; verdict is specific to this `(n,N)`."*

With τ (H1) already in hand, these two would let the tool **report a sound
generalisation certificate** for bounded-range local moves, and **honestly refuse**
it for VMMC (unbounded `r` in `N`) and for non-saturated runs. None of this is
implemented; it is a credible research path, with the VMMC `N`-case as the known
hard limit.

### 1.7 Per-example verdict (what generalises, and why)

| Example | Generalises in `n`? | Generalises in `N`? | Why |
|---|---|---|---|
| `single_metropolis` (`MAXD2=2`) | **Yes**, `n≥3` | **Yes** | atoms `{1,2}` saturated for all `n≥3`; single-particle, Metropolis ⇒ algebraic; relevant particles bounded by neighbourhood, not `N` |
| `metropolis_4x4` (`MAXD2=4`) | Yes, but only `n≥5` | Yes | `d²=4` atom needs `n≥4`; clean (no `L/2` tie) needs `n≥5`; a 3×3 PASS would **not** cover it |
| `kawasaki` | Yes (when saturated) | Conditional | swap is 2-site-local; but ergodicity is `N!`-restricted by design |
| `hop_8way_correct` | Yes | Yes | always-`D` pool (no variable-pool context), symmetric, local |
| `broken_*` | n/a (these FAIL) | n/a | a FAIL is a property of a realised context; it persists once realised, but absence on a too-small lattice can *hide* it (`broken_8way` needs `n≥4`) |
| `quadratic_field` | n/a | n/a | not translation-invariant (H1 fails); no generalisation claim |
| `vmmc_2d` | Plausible (saturated `n`) | **Not certifiable** | cluster reach unbounded in `N`; correctness is a construction theorem the checker cannot lift |

The single most important honest caveat: **a too-small lattice can hide a real
violation** (failure of saturation), so "smaller is safer" is wrong — for a trusted
verdict you must run the *smallest saturating* system, which for `MAXD2>2` or for
multi-particle contexts can be larger than 3×3.

---

## Part 2 — Can D4 be exploited like translation?

The lattice's full space group is `p4m = (translations) ⋊ D4`. Translation already
buys an `~n²` reduction via **orbit reduction**: BFS one representative per
translation orbit and *translate* its leaves. The question is whether the 8-element
point group `D4` (4 rotations + 4 reflections) buys a further `≤8×`.

### 2.1 Why D4 is fundamentally harder than translation

Orbit reduction over a symmetry `g` is sound iff the transition matrix is
**`g`-equivariant**: `T(g·s → g·t) = T(s→t)`. Translation-equivariance is *cheaply*
certifiable because **translation acts only on positions, not on the algorithm's
choices**:

- A displacement vector `δ` is translation-invariant. So the decision tree for `s`
  and for `g·s` is **literally the same tree** (same bits → same `δ` → translated
  successor), the leaf weights are translation-invariant, and successors simply
  translate. τ verifies this in **one** BFS.

`D4` acts on **both** positions **and** displacements (and on the canonical sort
order of the state). A rotation `R` sends `δ ↦ Rδ`, but `rand_choice!(rng, disps)`
picks the **same list index** from the **same list** regardless of `R`. So the
decision tree of `R·s` is the tree of `s` with the displacement list **permuted**
by `R` and the particle indices **re-sorted**. Equivariance `T(Rs→Rt)=T(s→t)` still
holds *after summing over the RNG* (a uniform choice over a `D4`-closed list is
`D4`-symmetric as a distribution), but it does **not** hold leaf-by-leaf. There is
no linear tag that exposes it from one BFS, because the obstruction lives in the
*permutation of the choice lists and the re-sorting*, not in the position algebra.

### 2.2 The dilemma the project keeps hitting

Two ways to get a sound `D4`-reduced **BFS**, both unsatisfactory:

- **Verify equivariance empirically:** BFS `R·s` for the generators of `D4` and
  check the leaves match `R·(leaves of s)`. But that BFSes ~3–8× per orbit — it
  *costs as much as the reduction saves*. (**"too slow"**.)
- **Assume equivariance** from a structural pre-check (every `disps` list is
  `D4`-closed; every predicate is `D4`-invariant — true since they go through `d²`;
  energy `D4`-invariant) and then reduce. Implementable, but the soundness now
  rests on a closure check over **arbitrary user-supplied lists** and on the claim
  that uniform-choice-over-a-closed-list is the *only* way the code consumes
  directional randomness. A subtle update (a non-uniform directional weight, a list
  that is `D4`-closed as a set but consumed order-dependently in a way that couples
  to a later branch) could break it and yield a **false PASS**. (**"requires
  trust"**.)

This is precisely the reported history: D4-for-BFS is either no faster or not
trustworthy. I believe that conclusion is **correct** and not a failure of effort —
it is structural. The `≤8×` upside does not justify a new false-PASS surface,
especially as the BFS is already cut `~n²` by translation and the *dominant* cost is
the symbolic BFS of the reps and the per-pair DB residual, not the state count.

### 2.3 The one honest win: D4 on the DB-check phase, verified on the *graph*

There is a place where D4 *can* help **soundly and without trusting the
algorithm**: the **detailed-balance check phase**, by exploiting symmetry of the
**already-computed transition graph** rather than of the code.

After the (translation-reduced) graph `T` is built, do two cheap, exact checks on
**the graph and energy themselves**:

1. **Energy `D4`-invariance:** `E(R·s) = E(s)` for all `s`, for `R` a 90° rotation
   and a reflection (generators). This is `O(#states)` symbolic comparisons of
   `LinForm`s. (For pairwise `d²` energies it always holds; for an absolute field
   it fails — and is correctly detected, exactly like τ.)
2. **Graph `D4`-equivariance:** for every edge `(s→t, w)`, check `(Rs→Rt, w)` is
   also an edge with equal weight, for the generators. This is `O(#edges)` hash
   lookups using a precomputed index permutation `s ↦ Rs`.

If both pass, then for any `R∈D4`:
`T(Rs→Rt)π(Rs) = T(s→t)π(s)` and `T(Rt→Rs)π(Rt) = T(t→s)π(t)`,
so **DB for `(Rs,Rt)` is *equivalent* to DB for `(s,t)`**. The DB residual then
needs to be evaluated on only **one pair per `D4`-orbit of pairs** — up to `8×`
fewer pair-checks — and the saving is **sound because we verified the property of
the concrete graph we are about to test, never a property of the code**. A state or
pair with a non-trivial `D4` stabiliser simply has a smaller orbit; the bookkeeping
(orbit reps, stabilisers) is standard and does not affect soundness.

Honest assessment of the payoff: this accelerates *only* the pair-residual phase
(for VMMC ≈ 2 s of ≈ 5 s warm), so the end-to-end gain is bounded (~1.3–1.8×), not
`8×`, because the BFS is untouched. It is nevertheless the **only** D4 exploitation
that is simultaneously fast-enough-to-bother and provably sound, and it is the angle
most worth implementing if D4 is revisited — *verify the graph, don't trust the
algorithm.* It also composes cleanly with the existing translation reduction (the
full `p4m` orbit of pairs).

### 2.4 A caveat even on the sound version

Verifying graph `D4`-equivariance requires the state set to be `D4`-closed (it is —
`R` permutes placements) and the index permutation `s↦Rs` to be exact (it is —
integer rotation on the torus). The reflection/rotation must be defined consistently
with the lattice (`(r,c) ↦ (c, n+1-r)` etc.) and composed with the canonical
re-sort. These are mechanical but must be got exactly right; a bug there would make
the *equivariance check* wrong, but note the failure mode is conservative: a wrong
permutation almost always makes the equivariance check **fail** (so we fall back to
all pairs), not silently pass. That asymmetry is what makes the graph-verified
approach safe in a way the code-trusting approach is not.

---

## Bottom line

- **Generalisation in lattice size and particle number is real but conditional.**
  Naive "all routes covered" is false — larger systems create new interaction
  distances, pool sizes, and arrangements. For **bounded-range, translation-
  invariant local moves** there is a sound theorem (locality + saturation), and the
  checker could deliver a generalisation *certificate* by adding two τ-style passes
  (particle-tagging for the locality radius; a geometric saturation check). The
  hard, probably-uncertifiable case is **`N`-generalisation of VMMC** (unbounded
  cluster reach); general universal generalisation is **undecidable** (Rice), so any
  certificate must live on the restricted, syntactically-checkable API class.
- **The smallest test system must be the smallest *saturating* one, not the
  smallest possible** — a too-small lattice hides violations. This deserves a
  prominent warning and an automatic saturation check.
- **D4 cannot soundly speed the BFS** without trusting the algorithm or paying the
  full cost; that long-standing conclusion is structurally correct. **D4 *can*
  soundly speed the DB-check phase** by verifying equivariance of the *computed
  graph* (not the code), for a bounded but real gain — the under-explored, honest
  option.
