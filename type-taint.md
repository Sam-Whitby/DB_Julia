# type-taint.md — can a τ-analogue certificate cut the τ-BFS over species orbits?

A deep, honest investigation of the **type-taint** idea: a tag on the particle
*species label* that — like `τ` for positions — would certify, from a **single
BFS**, that an algorithm is **species-equivariant**, so the τ-BFS could be reduced
over species (type-permutation) orbits as well as translation orbits. Because the
τ-BFS is the runtime bottleneck, this could in principle be a large speed-up.

All experiments are in [`doc/typetaint_poc.jl`](typetaint_poc.jl) (self-contained,
`julia doc/typetaint_poc.jl`). The findings below quote its output verbatim.

> **UPDATE — now IMPLEMENTED (supersedes the original recommendation below).** The
> original analysis judged the idea on `vmmc_2d` (which *declines* species — its
> sort tie-breaks on the type label) and concluded "do not adopt." Building
> `vmmc_2d_shuffle` — VMMC with a *random* candidate order, which is typical of real
> MC codes — produced a species-equivariant algorithm with a large tree whose τ-BFS
> the certificate cuts ~4–5× (56→12 reps, warm ≈6.3 s → ≈2.5 s). That tipped the
> cost/benefit, so the type-taint BFS reduction was implemented with τ-level rigor
> (parametric `Particle`, the `TypeTag` tag, a species-covariance guard, fail-loud
> fallback) and is validated to reproduce the direct-build graph exactly (AUDIT
> §5.13). The original §1–§7 below are kept as the design record; the soundness
> concerns they raise are addressed by the covariance guard, the fail-loud probe,
> and the graph-equality test rather than dismissed.

**Original TL;DR (pre-implementation): do not adopt the type-taint BFS reduction.**
The idea is sound *in principle* and the tag works and stays deterministic, but (1)
it does not help the actual bottleneck (VMMC *as then written*), (2) its soundness
rests on a no-unwrap discipline that is harder to guarantee for species labels than
for positions, and (3) the species symmetry it targets is already captured safely
and post-hoc by the Step-4 graph-verified reduction. *(Resolution: point 1 was
specific to the sorted VMMC; the typical randomised-order variant does benefit, and
points 2–3 are handled by the covariance guard + graph-equality validation.)*

---

## 1. Why it could matter

The τ-BFS reduces work over **translation** orbits: one representative per orbit is
BFS'd, and its leaves are translated to the rest — sound because `τ` certifies
translation-equivariance cheaply during that single BFS. The detailed-balance
*check* (Step 4) additionally quotients pairs by the spatial point group **and** by
species permutations, verified on the computed graph. But Step 4 is not the
bottleneck — the **BFS is** (VMMC: ~3 s of a ~4 s run). If species-equivariance
could be certified during the BFS the way `τ` certifies translation-equivariance,
the BFS could be reduced by up to `k!` (the species group) on top of translations.
For an all-distinct, type-rich system this is potentially huge (e.g. `S₄` = 24×).

## 2. The mechanism tested

`τ` works because a position is a wrapper (`TauNum`) that propagates a symbolic
offset and **flags** any *absolute*-position use (Mod, square, branch). The type
analogue: wrap the species label in a tag that permits only **equivariant** uses
and flags **absolute / order** uses:

```julia
struct Tt; v::Int; end
==(a::Tt, b::Tt)      = a.v == b.v          # allowed  (equality structure is σ-invariant)
hash(a::Tt, h)        = hash(a.v, h)        # allowed  (identity-based)
==(a::Tt, b::Integer) = (taint!(); a.v==b)  # FLAG     (compares to an absolute label)
isless(a::Tt, b::Tt)  = (taint!(); a.v<b.v) # FLAG     (imposes a label order)
+(a::Tt, b)           = (taint!(); a.v+b)   # FLAG     (arithmetic on a label)
Jc(a::Tt, b::Tt, d)   = …unwrap, build atom…# TRUSTED primitive: its internal a<=b
                                            #          canonicalisation is not a user branch
```

If a whole BFS runs **untainted**, types were used only through equality and atom
construction, which are σ-invariant, so the algorithm is species-equivariant.

## 3. What the experiments found

### E1 — the tag flags exactly the non-equivariant uses
```
alg_swap (equality only)      equivariant=true   tainted=false   -> tight
alg_move_type1 (==const)      equivariant=false  tainted=true    -> tight
alg_min_type (sort by type)   equivariant=false  tainted=true    -> tight
alg_energy (Jc only)          equivariant=true   tainted=false   -> tight
```
The tag is "tight" on these: clean ⟺ equivariant.

### E2 — SOUND except for the unwrap escape hatch (the critical hole)
```
alg_unwrap (reads t.v)        equivariant=false  tainted=false   -> *** UNSOUND ***
```
An algorithm that reads `t.v` directly (the raw `Int`) and branches on it dodges
every tagged operator, so it is **not flagged even though it is not equivariant**.
This is the same class of hole as `τ`'s `tau0`/`intval`, but **far more exposed**:
positions are used through a narrow geometry API, whereas species labels are
naturally used as plain integers everywhere, so "never unwrap a label" is a much
weaker contract than "never extract a raw coordinate". A *single* unwrap anywhere in
a translation → a false species-equivariance certificate → a wrong BFS reduction →
a **false PASS**. For a tool whose entire value is "no false PASS", that is the
decisive risk.

### E3 — deterministic
```
every algorithm: distinct taint verdicts over 1000 runs = 1
```
The tag introduces no nondeterminism: the verdict is a pure function of the code and
the (fixed, enumerated) BFS, exactly like `τ`. This requirement is met.

### E4 — fragility surface (which ordinary ops trip / bypass the tag)
```
t1 == t2                 taint=false   (ok, equality)
t1 != t2                 taint=false   (ok)
t1 == 1                  taint=true    (flagged, absolute)
t1 in Set([t1,t2])       taint=false   (ok, hash/==)
t1 in [1,2,3]            taint=true    (flagged, compares to Ints)
sort([t2,t1])            taint=true    (flagged, isless)
unique([t1,t1,t2])       taint=false   (ok, hash/==)
t1.v                     taint=false   (*** silent bypass — unwrap ***)
Dict(t1=>10)[t1]         taint=false   (ok, hash/==)
string(t1)               taint=false   (ok here; format-based logic could differ)
```
Two mitigating facts emerged. First, most *natural* idioms are intercepted
(`==`, `in`, `sort`, `unique`, `Dict`). Second, numeric operations on a tag that is
**not** an `Integer` subtype raise a `MethodError` — i.e. they fail **loud**, which
is safe (the user is forced to fix it, or it is a genuine flag). The only **silent**
unsound path is explicit field access `t.v`. So the hole is narrow, but it exists
and is the kind of thing a user does without realising (or that a helper they call
does on their behalf).

### E5 — the taint is DYNAMIC (and therefore less conservative, but still over-flags)
```
tiebreak on (pos, type)    [pos distinct]     equivariant=true  tainted=false
tiebreak on (parity, type) [parity collides]  equivariant=true  tainted=true
```
A pleasant surprise: the tag fires *per run* on type operations that **actually
execute**, not statically. Sorting by `(position, type)` with distinct positions
never reaches the type comparison (Julia's tuple `isless` short-circuits), so it is
**not** tainted — and is genuinely equivariant. This means the tag is *less*
conservative than a static "uses isless on a type" analysis would be.

But E5 also shows the flip side: when the tie-break *is* reached (parity collides),
the tag taints **even though the algorithm is still equivariant** (here the parity
key decides the outcome, so the type tie-break is immaterial). So the tag remains
**conservative** — it declines some genuinely-equivariant algorithms. This is the
VMMC pattern (its cluster builder tie-breaks candidate order on the species label),
and it is exactly why the Step-4 graph verification already declines VMMC's species
symmetry.

## 4. Soundness argument (and its boundary)

*Claim.* If the BFS of a representative `s` runs untainted, then for every species
permutation `σ`, `leaves(σ·s) = σ·leaves(s)`.

*Why.* Untainted means every use of a label was an equality test between labels or
an atom construction. Equality structure is σ-invariant and `σ·s` has identical
positions (σ relabels types only), so **every branch is taken identically** for `s`
and `σ·s`; atoms are relabeled by `σ`; outputs carry types with particles. Hence the
leaf multiset of `σ·s` is the σ-relabel of that of `s`. ∎

*Boundary.* The argument assumes labels are touched **only** through tagged
operations. The `t.v` unwrap (E2/E4) breaks the assumption silently. So soundness =
"tag intercepts every label use" + "no unwrap" — a discipline. For `τ` the same kind
of discipline is acceptable because positions flow through a small, audited API; for
species labels the discipline is much harder to keep, which is the core problem.

## 5. Does it help the bottleneck? (Mostly no.)

- **VMMC (the bottleneck): no clean benefit.** Its candidate sort tie-breaks on the
  species label; whenever that tie-break is reached it taints (E5), so the BFS would
  not be species-reduced — the same conservative decline the shipped Step-4 check
  already makes. A *dynamic* tag might salvage a *partial*, per-representative
  reduction for those VMMC states where the tie-break never fires, but that is
  uncertain, fiddly, and at best shaves a fraction of the BFS.
- **The cases that do benefit are mostly already fast.** Single-particle Metropolis,
  Kawasaki, etc. are species-equivariant, but their BFS is already ~0.03–0.2 s — the
  `k!` reduction there saves milliseconds.
- **Genuine win only for type-rich *large* systems** (many distinct species on a big
  lattice, e.g. `S₄`=24× on a 4×4), which are rare and already strained by the
  exhaustive state count. There the BFS reduction could matter — but see §6 for a
  safer way to get it.

## 6. Cost, and safer alternatives that get most of the benefit

**Implementation cost of the tag is high and invasive.** `Particle.t` would become a
tagged value, so the struct, `norm_state` (which puts the type into the integer
`CState`), `Jc`, and the energy evaluated on *concrete* (`Int`) states would all
have to be tag-generic, and the whole engine would carry a second soundness surface.

**Safer alternatives, in order of preference:**

1. **Keep the Step-4 graph-verified species reduction (already shipped).** It
   captures the *same* species symmetry, verified on the *computed graph* up to the
   atom relabeling — sound, post-hoc, no API change, no unwrap risk. It only misses
   the *BFS* speed-up, which (per §5) the bottleneck does not get anyway.
2. **Verify species generators by re-BFS (sound, no fragility).** To reduce the BFS
   over species, BFS a representative *and* its species-generator images, and check
   `leaves(σ·s) = σ·leaves(s)` directly; trust composition for the rest of the orbit.
   This needs `1 + #generators` BFS per combined orbit instead of `|group|`, so it is
   a *modest* (~2× for `S₃`) sound win with **no** tag, **no** API change, and **no**
   unwrap hole — and it auto-declines VMMC (the generator re-BFS reveals the
   tie-break mismatch). This is the right tool if a type-rich large system ever needs
   it.
3. **Tag + differential fuzzing**, only if 1–2 are insufficient: ship the tag but, on
   every run, additionally re-BFS one species-generator image of one representative
   and assert it matches the tag's prediction. That converts the unwrap hole from a
   silent false PASS into a loud mismatch — but it reintroduces a re-BFS cost and is
   strictly worse than (2) for soundness-per-cost.

## 7. Recommendation

**Do not implement the type-taint BFS reduction.** It is an elegant τ-analogue and
the PoC shows the mechanism is real and deterministic, but:

- it does **not** speed the actual bottleneck (VMMC declines, like Step 4);
- the cases it speeds are already fast; only rare type-rich large systems would gain;
- its soundness depends on a "never unwrap a label" discipline that is **much weaker
  in practice** than `τ`'s position discipline, and a single violation is a false
  PASS — unacceptable for this tool's guarantee;
- the species symmetry it targets is **already captured safely** by the shipped
  Step-4 graph-verified reduction, with no API change and no new soundness surface.

If a type-rich large system ever makes the species BFS reduction worthwhile, use
**alternative 6.2 (verify generators by re-BFS)** rather than the tag: it is sound by
construction, needs no `Particle` surgery, and fails loud. The dynamic-taint finding
(E5) is worth remembering — any future tag should be a *runtime* tag (fires only on
executed label ops) to avoid over-declining — but the unwrap exposure keeps it below
the project's correctness bar today.
