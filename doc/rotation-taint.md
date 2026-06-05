# rotation-taint.md — can rotation / reflection join the τ-BFS via a taint trick?

A deep, honest investigation, prompted by the success of the **species type-taint**
([`type-taint.md`](../type-taint.md), now shipped): if a discrete symmetry
(species permutation) can be brought *into* the BFS by a taint that certifies
equivariance from a single BFS, can the **lattice point group** — 90° rotations
and reflections (the `D4` of the square torus, the rotational part of `p4m`) — be
brought in too, to cut the number of states BFS'd (the bottleneck) by up to `|D4| = 8×`?

All experiments are in [`doc/rotation_taint_poc.jl`](rotation_taint_poc.jl)
(self-contained, `julia doc/rotation_taint_poc.jl`). Findings quote its output verbatim.

> **TL;DR (under the CURRENT contract) — do NOT implement a rotation/reflection
> BFS reduction. It is provably impossible to do soundly** *as long as directions
> are hardcoded constants inside the algorithm.* The point group then has **no
> single-BFS equivariance certificate**, and verifying it requires BFS-ing the very
> states one would hope to skip. The post-hoc Step-4 graph check is the correct —
> and only sound — place for `D4`, and it is already there.
>
> **UPDATE — a CHANGED contract restores it (see §7).** If the direction set is
> *supplied* to the algorithm (a params-file contract) and declared **closed under
> the point group**, directions become covariant objects the group *permutes* —
> structurally identical to species labels. Then a single clean BFS of the rep
> **does** certify rotation-equivariance and the orbit can be derived soundly,
> validated in [`doc/rotation_taint_contract_poc.jl`](rotation_taint_contract_poc.jl).
> This is a real, implementable win (up to `|D4|=8×`, stacking with
> translation×species) for the class of *isotropic* algorithms — at the cost of an
> API/contract change and a more complex position tag. The "impossible" result
> below is specific to the hardcoded-direction contract.

---

## 1. Why translation and species *can* be certified from one BFS

The two shipped single-BFS certificates work for the **same structural reason**:
the symmetry is realised by **identical control flow**, so one BFS of the
representative sees it, and **every** state is reached from its rep by **one**
certified group element — no un-verified intermediate state is ever trusted.

| symmetry | how it acts | why one BFS suffices |
|---|---|---|
| **translation** (τ) | one continuous **offset** added to every particle | output position is a pure `+offset` shift (`is_covariant_pos`: `cr=1,cc=0/…`); a single symbolic BFS proves the offset cancels in all weights → equivariant for **all** translations at the rep |
| **species** (`TypeTag`) | a **relabel** used only via `==`, `hash`, atom-build | a single tagged BFS that stays clean means control flow never saw an absolute label → equivariant for **all** `σ` at the rep |

In both cases the per-rng-path map is the **identity**: with a shared rng the
algorithm makes the *same* decisions on `s` and on `g·s`, and the output is the
transformed output. That is exactly what a value-taint can observe in one pass.

## 2. The obstruction: rotation is realised by an rng-outcome *permutation*

A rotation **mixes coordinates** ((r,c) → (c, N+1−r)) and is **discrete**, so there
is no continuous offset to cancel. Worse, the canonical MCMC move — *enumerate a
fixed set of directions and pick one* — is rotation-equivariant **only at the level
of the leaf multiset**, achieved by a *permutation of rng outcomes*, never by
identical control flow:

```
E2 — per-rng-path correspondence FAILS under rotation (the obstruction)
   g.(North . s)  = [(2, 4, 1)]
   North .(g.s)   = [(1, 3, 1)]
   identical per-path?  false   <- a tau/species-style certificate needs YES
   => a single-BFS symbolic/control-flow taint DECLINES this equivariant move.
```

With a shared rng, "move North" on `s` followed by rotating is **not** "move North"
on the rotated `s`: the rng path *North on `s`* corresponds to a *different* path
(*East on `g·s`*). A τ-style symbolic taint or a species-style control-flow taint
sees the divergence and would **decline** a perfectly rotation-equivariant move.

The same kills the "rotation **matrix** taint" idea (an abstract `R` with `R⁴=I`
carried on positions): when the algorithm adds a *fixed* offset `(1,0)`, the output
is `R·pos + (1,0)`, which is not `R`-covariant, so it taints — unless the taint
recognises that the *set* of enumerated offsets is closed under `R` with matching
weights. But that is a property of the **branch set**, not of any value flowing
through an operator; recognising it is exactly "compare the leaf multiset to its
rotation" — which needs the rotated state's behaviour (§4), not a one-pass taint.

## 3. The symmetry is nonetheless real — at the multiset level

```
E3 — leaf MULTISET is equivariant for the isotropic move (Mechanism C basis)
   rot90    : leaves(g.s)==g.leaves(s) for all samples : true
   rot180   : leaves(g.s)==g.leaves(s) for all samples : true
   rot270   : leaves(g.s)==g.leaves(s) for all samples : true
   refl     : leaves(g.s)==g.leaves(s) for all samples : true
```

So `D4`-equivariance **does** hold for isotropic algorithms and **is** checkable —
but the check is `leaves(g·s) == g·leaves(s)`, a comparison of the rep's leaves
against the **rotated state's** leaves. And that comparison is **tight**: it accepts
exactly the right subgroup, including chiral algorithms that have `C4` but not
reflection symmetry — the same subgroup discovery Step-4 already performs on `p4m`:

```
E4 — Mechanism C is tight & discovers the correct subgroup
   iso       equivariant under: rot90, rot180, rot270, refl
   drift     equivariant under: (none)
   absolute  equivariant under: (none)
   chiral    equivariant under: rot90, rot180, rot270        # C4 only, NOT reflection
```

## 4. The soundness wall: you cannot *skip* a state

Because there is no single-BFS certificate (§2), the only way to "verify" rotation
is the §3 comparison, which needs `leaves(g·s)` — i.e. **BFS-ing `g·s`**. The
tempting shortcut — *verify the generators at the rep, then derive the rest of the
orbit by composition* — is **unsound**: an algorithm equivariant at the rep can
fail at a derived orbit member, and the derivation never executes that member:

```
E5 — deriving an un-BFS'd state by rotation is UNSOUND (the wall)
   generator check at rep {(1,1)} passes : true
   r180.rep = [(4, 4, 1)] (the trap site)
   derived == truth (would be sound) : false
   => skipping r180.rep records WRONG transitions -> risk of false PASS.
   => the ONLY sound check BFSes r180.rep itself: no BFS saved.
```

Composition fails because `leaves(σ₂σ₁·rep) = σ₂·leaves(σ₁·rep)` needs the generator
relation **at the intermediate state `σ₁·rep`**, which verifying only *at the rep*
never establishes. (This is exactly why the shipped **species** reduction relies on
the *tag* — which certifies **all** `σ` at the rep, with every state reached by a
single `σ` — and **not** on the "verify generators + trust composition" recipe
floated as alternative 6.2 in `type-taint.md`; E5 shows that generic recipe has a
soundness gap for any discrete group lacking a per-state certificate.)

Consequently the **sound** BFS reduction obtainable from `D4` is exactly `1.00×`,
no matter how inviting the orbit counts look:

```
E7 — K=3 on 4x4 torus : orbit counts
   translation reps        (BFS today) : 35
   translation x D4 reps   (hoped-for) : 10
   hoped-for naive factor              : 3.50x  (UNREACHABLE soundly -- see E5)
   sound BFS reduction from D4         : 1.00x (none)
```

The check is deterministic (E6), but determinism was never the issue — soundness is.

## 5. The one conceivable single-BFS certificate, and why it is not worth it

A single-BFS rotation certificate *could* exist only by **redesigning the move
API** so that the algorithm cannot express a non-equivariant move: positions would
flow through a rotation-tagged geometry primitive (like τ's covariant-position
gate), and moves could *only* be drawn from offset sets **declared** to be closed
under `D4`, applied with equal weight. Then a clean BFS of the rep would certify
`D4`-equivariance the way the covariant-position gate certifies translation. But:

- it is an **invasive API change** that forces every algorithm to be written in a
  rotation-covariant style;
- it **shrinks expressiveness** — the most common real move (pick a direction by an
  rng index and use it) is not in this form, and anisotropic/chiral/absolute moves
  are excluded outright;
- it certifies only what is *already* free post-hoc: energies and acceptances are
  isometry-invariant anyway (functions of `pbc_d2`), and Step-4 already discovers
  the exact point-group subgroup on the computed graph (E4) with **no** API change,
  **no** new soundness surface, and correct chiral/subgroup handling.

So even the one path to a single-BFS certificate duplicates an existing capability
at a large expressiveness and complexity cost.

## 6. Recommendation

**Do not implement a rotation/reflection BFS reduction.** It is not a tuning
question — it is **provably impossible to do soundly** for a tool whose contract is
"never a false PASS":

- there is **no single-BFS certificate** for the point group (§2), because the
  symmetry is an rng-outcome permutation, not identical control flow;
- empirically verifying it requires BFS-ing the candidate-skip states (§4), so the
  sound BFS saving is **zero** (§4–§7);
- the symmetry **is** already captured, soundly and tightly, post-hoc by the
  **Step-4 graph-verified `p4m` reduction** (which discovers the right subgroup,
  E4) — exactly where a graph-level multiset comparison belongs.

**Difficulty / time estimate for any future attempt:** the only variant that would
*save* BFS work (generator + composition) is unsound (E5) → must not ship. A
*sound* variant that BFSes every state and merely checks `D4` earlier yields **no
speed-up** (it just relocates Step-4 work) at real implementation cost → negative
value. The single-BFS-certificate route (§5) is a multi-week API redesign that
narrows expressiveness and still only re-earns the existing Step-4 result.

**If the BFS bottleneck needs to come down** *without changing the contract*, the
levers are the shipped translation×species reduction, `-parallel`, depth/branch
pruning, or interning — not the point group. But the contract **can** be changed —
see §7, which is the right way to actually get the point group into the BFS.

---

## 7. Supplied directions change the contract — and the conclusion

> **STATUS — now IMPLEMENTED.** The supplied-directions point-group BFS reduction
> described in this section is shipped in `dbc.jl` (the `DirTag`/`move`/`rand_move!`/
> `rev` contract, the `pointgroup_subgroup` static closure check, and the
> translation×species×point-group orbit reduction in `build_transitions`). Examples
> that declare `const MOVES` opt in; the checker reports the certified subgroup and
> reduces the τ-BFS accordingly. Measured wins (warm BFS): single_metropolis 56→4
> reps, vmmc_2d_shuffle 12→4 reps (~2.5 s → ~0.8 s), vmmc_2d → D4-only 56→8 reps
> (~2.5 s → ~0.7 s), horizontal_metropolis discovers exactly D2. Soundness is
> validated by a graph-equality test against a direct all-states build (AUDIT
> §5.14). The remaining-problems in §7.1 were handled exactly as proposed (trusted
> primitives, fail-loud probe, covariance via the existing τ gate, graph-equality
> validation).


The whole §1–§6 obstruction rests on one thing: a **hardcoded** offset like `(1,0)`
is an **absolute constant**. Under rotation it does *not* transform, so it cannot be
"relabeled", and the per-rng path `move (1,0)` on `s` has no covariant counterpart
on `R·s` (E2). That is exactly why rotation lacked the *identical-control-flow*
property that lets τ and species certify their orbits from one BFS.

**The fix the user proposed: stop hardcoding directions — *supply* them.** Make the
direction set `D` a declared object (a params-file contract), require moves to be
written as `pos + d` with `d ∈ D`, and put the rotation-taint on **both** positions
and directions. Now a direction is no longer an absolute constant: it is a covariant
object that the point group **permutes**. This makes rotation structurally identical
to species:

| coordinate | acted on by | relabeled object | branch set preserved because |
|---|---|---|---|
| species | `σ ∈ Sₖ` | atom labels | `σ` is a bijection on labels |
| **direction** | `R₀ ∈ D4` | the **supplied set `D`** | `R₀(D)=D` (**static** closure check) |

Once directions are relabelable, a single clean BFS of the rep certifies rotation
the same way species is certified — and crucially the soundness comes from a
**static** property of the params (`D` closed under the group), provable **without
executing `R·s`**:

```
leaves(R·s) = { R(p) + d  : d ∈ D }
R·leaves(s) = { R(p) + R₀(d) : d ∈ D } = { R(p) + e : e ∈ R₀(D)=D }   ⟹  equal multisets
```

`doc/rotation_taint_contract_poc.jl` verifies the whole claim:

```
C1 — taint verdict  vs  derivation soundness
   algorithm  clean?   derivable?   match?
   iso        true     true         OK
   drift      false    false        OK         # anisotropic weight (dir identity) -> flagged
   rowonly    false    false        OK         # absolute axis choice            -> flagged
   trap       false    false        OK         # absolute coordinate (E5 trap)   -> flagged
   => clean <=> soundly-derivable : the taint accepts EXACTLY the right algos

C3 — the clean ISO algorithm: single-BFS derivation reproduces direct BFS
   R.leaves(rep) == leaves(R.rep) for all reps, all R in D4 : true
   (NO execution of R.rep needed -- certified from rep's own BFS + D-closure)
```

The E5 trap — the case that made the old approach unsound — is now **flagged** (it
reaches for an absolute coordinate), so it is declined, never derived. The taint is
**tight**: clean ⟺ soundly-derivable.

### 7.1 Remaining problems (real, but bounded)

This is no longer "impossible" — it is a genuine engineering trade-off:

1. **Positions must become opaque covariant 2-vectors.** Today's geometry does
   per-coordinate arithmetic (`p.r - q.r`). Under rotation, a single coordinate is
   *not* rotation-invariant, so per-coordinate access must be banned in user code and
   confined to **trusted primitives** (`dist2`, `same_site`, `move`) — the same
   "trusted primitive + no-unwrap discipline" used for species (`Jc`) and τ, but a
   **larger surface** because positions are used far more richly than labels.
2. **A new, richer position tag.** `TauNum` only tracks an additive offset (`v + cr·τ_r
   + cc·τ_c`); rotation couples `r` and `c`, so the tag must be a covariant 2-vector
   under the affine group (point-group linear part **+** translation offset), with a
   covariance gate generalising `is_covariant_pos`. This is the bulk of the work and
   the main soundness-surface growth.
3. **A contract/API change for users.** Directions move to a params file; moves are
   written `move(pos, d)`; hardcoded offsets are rejected. Existing examples need
   rewriting (or a dual path: only offer the rotation reduction when a direction set
   is declared, leaving the current path untouched).
4. **Class boundary.** Genuinely anisotropic algorithms (axis-dependent coupling,
   external fields, chiral-only moves) correctly **decline** — sound, but it bounds
   the win to *isotropic* moves (a large class: single-particle/spin Metropolis,
   Kawasaki exchange, isotropic cluster moves).
5. **Same residual hole as species, handled the same way.** Soundness rests on the
   no-unwrap discipline; mitigate exactly as species does — a covariance guard on
   outputs, fail-loud on un-overloaded ops, and a **graph-equality validation test**
   (`rotation=true` build vs a direct build, like `_species_graph_consistency`).
   So the rigor level is the *same as the shipped species reduction*, on a bigger
   surface.

### 7.2 Recommendation, difficulty, expected speed-up

**Recommendation: worth implementing as an opt-in, staged on top of the existing
combined-orbit BFS — but only if isotropic algorithms are a priority.** Engage the
point-group reduction **only** when an example declares a closed direction set;
otherwise behave exactly as today. Gate the trust behind the graph-equality
validation test before relying on it.

- **Expected speed-up:** up to `|point group|` (8× for D4, 4× for C4), stacking
  *multiplicatively* with translation×species. Realistically ~2–4× on the small
  systems here (stabilisers shrink it — cf. E7: 35→10 translation×D4 reps for K=3),
  more on larger / low-symmetry systems. This is a bigger lever than species for
  isotropic models and directly hits the BFS bottleneck.
- **Difficulty:** moderate-to-high — **larger than the species type-taint work**,
  dominated by item 2 (the covariant-2-vector position tag) and item 1 (geometry-API
  rewrite). Roughly 1.5–2.5× the species effort. Recommended order: (a) params-level
  direction set + static closure check (cheap, isolated); (b) the covariant position
  tag + trusted geometry primitives behind a feature flag; (c) extend
  `combined_trep_orbits` to translation×species×point-group and derive leaves by
  translate∘relabel∘rotate; (d) the graph-equality validation test + decline/subgroup
  tests; (e) rewrite examples to the contract.

The transferable lesson, now complete: a discrete symmetry can enter the BFS **iff**
its action is realised as a *relabeling of covariant objects the algorithm uses
opaquely*. Species labels already are such objects; **directions become such objects
the moment they are supplied rather than hardcoded** — which is exactly why the
contract change works.
