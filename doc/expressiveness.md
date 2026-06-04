# Expressiveness of the DB_Julia checker — a definitive scope report

What classes of weight, energy, and condition functions can the checker verify
*exactly*, what is outside its reach (and **why**), and how could each boundary be
pushed? This is a complete, code-grounded account.

The short version: the checker is exact precisely because everything it touches
lives in one of two finitely-decidable worlds —

- **weights** = *Laurent polynomials in `exp(−β·linear)` monomials, over ℚ,
  divided by products of `(1 − exp(−β·linear))`*; and
- **parameter space** = carved by **linear** inequalities into polyhedral
  **chambers**, decided by an exact rational LP.

Everything that is "handled" reduces to these; everything "not handled" leaves one
of them. Crucially, anything outside is a **hard error (`CantHandle`), never a wrong
answer** — and most exotic functions cannot even be *written* through the API.

---

## 1. The exact representable class (precise statement)

Let `J = (J_a)` be the coupling atoms (`couplingJ[t1,t2,d²]`, canonical `t1≤t2`,
and named field params like `:fieldH`). The checker is exact for:

| Object | Allowed form | Code type |
|---|---|---|
| **Energy** `E(s)` | `⟨c(s), J⟩`, **linear** in `J`, `c(s) ∈ ℚ^{|A|}` | `LinForm` (linear) |
| **Boltzmann exponent** | `⟨L, J⟩`, **linear**, `L ∈ ℚ^{|A|}` (½, ⅓… all fine) | `ThBoltz(RatForm)` |
| **Weight value** | `num / ∏_k B_k(J)`, `num = Σ_v p_v(J)·exp(−β⟨v,J⟩)`, `p_v ∈ ℚ[J]`, each `B_k` a 1–2-term constant-coeff binomial in exp-monomials | `Val = (BSum, [BSum])` |
| **Threshold combinators** | `+ − × ÷`, `min`, `max`, piecewise, `linear` — `÷` by any binomial (`1−exp`, `1+exp`, …) | `ThOp, ThMin, ThMax, ThLinear, ThPiece` |
| **Branch condition** | `⟨h, J⟩ {<, ≤} 0`, **linear** hyperplane, `h ∈ ℚ^{|A|}` | `Cond(RatForm, strict)` |
| **Constants** | **rational** `ℚ` only | `ThConst(Q)` |

`BSum = Dict{ExpVec, Poly}` with `ExpVec = Vector{Q}` and `Poly = Dict{Mono, Q}`:
the numerator is a multivariate **Laurent polynomial** in the exp-monomials, whose
**coefficients are themselves polynomials in the couplings** (`ℚ[J]`), with
rational-linear exponents. `Q = Rational{Int128}` (overflow-checked → fail-loud).
*(Before the Tier-2 work the coefficients were bare rationals and denominators were
restricted to `∏(1−exp); both are now generalised — see §4.)*

### Why this is exact (the two load-bearing facts)

1. **Distinct exponent vectors give linearly independent exponentials.** On any
   open chamber, an exponential polynomial `Σ_v p_v(J)·exp(−β⟨v,J⟩) ≡ 0` **iff**
   every coefficient polynomial `p_v ≡ 0` (and a polynomial vanishing on an open set
   is identically zero, i.e. all its monomial coefficients are zero in ℚ). The DB
   residual is reduced (denominators cleared) to exactly such a sum and tested this
   way. This is *the* reason the verdict is exact and complete.
2. **Linear conditions ⇒ polyhedral chambers ⇒ exact LP.** The branch hyperplanes
   cut `ℝ^{|A|}` into open chambers; adjacency is connected, so a sign-flip BFS with
   an exact rational simplex enumerates them all.

Every extension below is feasible exactly to the extent it preserves *(1)* a
coefficient/exponent ring with **decidable equality and a linear-independence
basis**, and *(2)* a parameter-space decomposition we can **decide cell-nonemptiness**
for.

---

## 2. The constructibility gate (you often cannot even write the exotic thing)

The public API is the first filter. Atoms appear inside `exp` (`th_boltz`), inside
conditions (`c_lt/c_le`), and now — as bare polynomial values — inside `th_linear`;
constants are `ℚ`; exponents and conditions take `LinForm` (linear). Consequences,
before any runtime check:

- You **cannot construct** a non-linear exponent: `th_mul(th_boltz(L1),th_boltz(L2))`
  yields `exp(−β(L1+L2))` (exponents **add**), never `exp(−β·L1·L2)`.
- You **CAN now construct** a bare polynomial-in-`J` weight via `th_linear` and
  `th_mul` (e.g. `th_mul(th_linear(J11), th_linear(J22))` = `J11·J22`); coefficients
  live in `ℚ[J]`.
- You **cannot construct** a non-linear condition (`J11² > J22³`): `c_lt/c_le` take
  `LinForm`.
- You **cannot construct** `log`, `√`, `^`, trig, or an irrational `th_const`: there
  are no such builders. (`max` now exists via `th_max`.)

So most "exotic" requests are blocked at authoring time. The runtime gates
(`val_div`, `min_condition`, `eval_static`) catch the rest and `CantHandle`.

---

## 3. Definitive capability table

| Function / construct | Verdict | Why | Mitigation (difficulty) |
|---|---|---|---|
| Laurent polynomials in `exp(−β·linear)` | **✅ full** | the native algebra | — |
| Half/any-rational exponent `exp(−βΔE/2)` | **✅ full** | `ExpVec ∈ ℚ` | — |
| Products/sums of exps, exact cancellation (VMMC ratio) | **✅ full** | `bs_mul/bs_add` add exponents, cancel zeros | — |
| `min(1, exp)`, `min(ratio, 1)` with **single-hyperplane** switch | **✅** | `min_condition` → balanced binomial → hyperplane | — |
| Piecewise on **linear** conditions | **✅** | `ThPiece` + LP chambers | — |
| Denominator `∏ (1 − exp(−β·linear))` | **✅** | `Val.den` multiset; nonzero on open chambers | — |
| Many-body coupling `J[a,b,c]` (3+ body) | **✅** (needs atom ctor) | still **linear** in its own atom | add atom constructor (easy) |
| `exp(−β·\|ΔE\|)` | **✅ via piecewise** | `th_piece` on `c_lt(ΔE)` → `exp(∓βΔE)` | author it as 2 branches |
| `max(a,b)` (single-hyperplane switch) | **✅ (`th_max`)** | mirror of `th_min`, complementary branch | — (implemented) |
| Denominator `(1 + exp)` (Barker/Glauber accept) | **✅ (`val_div`)** | binomial factor stored explicitly; `1+exp>0` | — (implemented) |
| Denominator `(2 − exp)`, `(1 − 2·exp)`, any 1–2 term const-coeff binomial | **✅ (`val_div`)** | residual clears it exactly | — (implemented) |
| `min` of multi-term exp-sums (switch not a hyperplane) | **❌** | switch is a transcendental surface | transcendental cells (hard/open) |
| `min` over condition-bearing operands / nested `min` | **❌** | `eval_static` errors | recursive chamber split (moderate) |
| **Non-exp polynomial weight** `J11·J22`, `⟨L,J⟩²` | **✅ (`th_linear` + ℚ[J] ring)** | coeffs are now polynomials in `J` | — (implemented) |
| **Non-linear exponent** `exp(−β·J11·J22)` | **❌ (unconstructible)** | `ExpVec` is linear | polynomial exponents + (usually) CAD (hard) |
| **Non-linear condition** `J11² > J22³` | **❌ (unconstructible)** | conditions linear | CAD (hard, doubly-exp) |
| **Commensurate trig field** `H·Σcos(2πr/L)` | **❌** | coeff would be algebraic, not ℚ | cyclotomic number field (hard, sound) |
| Incommensurate trig / `cos(√2·r)`, `log`, general `√(sum)` | **❌ fundamental** | no exact finite representation | none (irreducibly transcendental) |
| Irrational constant accept prob (e.g. `1/√2`) | **❌** | `ThConst` is ℚ | algebraic constants (moderate) |
| Continuous RNG value (Gaussian/continuous displacement) | **❌ fundamental** | BFS enumerates discrete bit-trees | discretise the move (changes algorithm) |
| Unbounded "resample-until-valid" loop | **❌ (maxdepth)** | infinite decision tree | rewrite as "enumerate-valid-then-choose" |

---

## 4. Detailed analysis of the categories you asked about

### 4.1 Polynomial expressions

- **In the exp-monomials: fully handled.** The numerator is an arbitrary Laurent
  polynomial; products multiply (exponents add), sums add, and zero coefficients
  cancel exactly. There is no degree limit and no loss.
- **As bare functions of `J` (non-exp), e.g. `J11·J22`, `(J11−J22)²`: NOW HANDLED
  (Tier 2, implemented).** The coefficient ring was widened from ℚ to **ℚ[J]**
  (`Poly`, multivariate polynomials in the atoms), and the builder **`th_linear(L)`**
  injects the bare value `⟨L,J⟩`; products of `th_linear`s give higher degree. The
  DB residual clears denominators to `Σ_v p_v(J)·exp(−β⟨v,J⟩)`; by fact (1) each
  `p_v(J)` must vanish on the open chamber, and a polynomial vanishing on an open set
  is **identically zero**, i.e. all its monomial coefficients vanish — a finite exact
  ℚ-test (`isempty(res)` after the `Poly` coefficients cancel). Constant coefficients
  take a fast path so existing cases are unaffected. See `examples/poly_rate_accept.jl`
  (a rate factor `a` × Boltzmann exponentials).

### 4.2 Denominator structure (NOW GENERAL — Tier 1, implemented)

`Val.den` now stores each binomial factor **explicitly** (as a small `BSum`),
rather than just the `L` of a `1−exp(−βL)`. `val_div` accepts any 1- or 2-term
exp-polynomial with **constant (rational) coefficients** as a denominator:
`1−exp` (VMMC), **`1+exp` (Barker/Glauber, `examples/barker_accept.jl`)**, `2−exp`,
`1−2·exp`, a single exp-monomial, etc.

A subtle but important soundness point: clearing the denominator is sound *whatever
the binomial*, not only sign-definite ones. The DB residual `R = N/D` is multiplied
through by the common denominator and `N` is tested for being identically zero;
since the physical transition probabilities are continuous everywhere (the
individual `(1−exp)` factors cancel within each transition's leaf-sum), `N ≡ 0` ⟺
`R ≡ 0` on the chamber ⟺ detailed balance — independent of where `D` vanishes. So
`1+exp` (never zero) and `1−exp` (zero on a hyperplane) are equally sound. The only
restriction kept is "1–2 terms, constant coefficients", i.e. the engine represents
*binomial* denominators; a non-constant or larger denominator still `CantHandle`s.

### 4.3 `min` / `max`

- **`th_max` is now implemented** (Tier 1): it shares `min`'s `a<b` hyperplane and
  selects the complementary branch.
- `min`/`max` are sound **only when the switch is a single hyperplane**. VMMC's
  `min(ratio,1)` works because the cleared difference is a *balanced binomial*
  `c·(exp(−βL1) − exp(−βL2))`, whose sign is `sign(L1−L2)` — a hyperplane. A `min`
  of two **multi-term** exp-sums has a switch surface `Σexp < Σexp` that is **not** a
  hyperplane (it is transcendental in `J`); `min_condition` correctly `CantHandle`s
  it. Deciding such exp-sum inequalities exactly is hard in general (it brushes
  Schanuel-conjecture territory), so this is the natural hard ceiling for `min`.
- `min`/`max` operands may now be condition-free polynomial/exp expressions (a
  `min_condition` with a non-constant coefficient is rejected — the switch would not
  be a hyperplane). Truly **nested** `min`-of-`min` (a piecewise switch) still errors
  in `eval_static` (Tier 2 #6, partial). **Remaining mitigation:** recursive
  chamber-splitting of the operands; moderate effort, sound.

### 4.4 Non-linear coupling terms

This is the crux of several of your examples. Note first a clarification:
**many-body interactions are *not* non-linear couplings.** A 3-body term
`J[a,b,c]·(count)` is *linear* in the (new) atom `J[a,b,c]`; only a new atom
constructor is needed. Non-linearity means **products of distinct couplings**:

- **`exp(−β·J11·J22)` (non-linear exponent):** not constructible; `ExpVec` is linear.
  *Mitigation:* allow exponents to be **polynomials** in `J`, grouping monomials by
  canonical polynomial form. Fact (1) still gives independence for distinct
  exponent-polynomials, so the residual test survives — **but** any *condition*
  touching such an exponent (a `min` switch, a `≤`) becomes non-linear, which forces
  the chamber machinery to CAD (below). Feasible only in the special case of
  polynomial exponents with **no conditions on them**; otherwise hard.
- **`J11·J22` as a plain weight (non-exp, non-linear): NOW HANDLED** by the ℚ[J]
  coefficient ring of §4.1 (`th_mul(th_linear(J11), th_linear(J22))`).
- **`J11² > J22³` (non-linear condition):** not constructible; conditions are linear.
  *Mitigation:* replace the polyhedral chamber engine with **Cylindrical Algebraic
  Decomposition** over the polynomial conditions — exact and complete, but
  doubly-exponential in the number of atoms and a large implementation. This is the
  only fully-general route, and it is the dominant cost wall: it abandons the cheap
  exact LP that pillar (2) relies on. For small atom counts it is tractable in
  principle; for the systems studied here it would dominate runtime.

### 4.5 Trigonometric / field functions

A field enters as `E += H·κ(s)` where `H` is an atom and `κ(s)` is a per-state
spatial sum. Today `κ(s) ∈ ℚ` (e.g. `Σ r²`). A trig field `κ(s) = Σ cos(2π r/L)`
makes `κ(s)` an **algebraic** number.

- **Commensurate angles (`2πk/L`):** the values live in the **cyclotomic field
  ℚ(ζ_L)**, which has exact arithmetic and decidable equality. Generalising `Q`,
  `ExpVec`, and `BSum` coefficients to elements of a fixed number field would make
  such fields exact: the monomial grouping keys on the algebraic exponent value, and
  linear independence of `exp(−βH·κ)` holds for distinct `κ`. Sound, but a deep
  refactor (the numeric type `Q` threads through everything). Difficulty: hard.
- **Incommensurate / genuinely transcendental angles (`cos(√2 r)`):** no exact finite
  representation; equality of exponents becomes undecidable. **Out of reach in
  principle**, not just in implementation.

### 4.6 Other boundaries worth stating

- **Absolute values** inside exponents: handle via piecewise on the sign hyperplane
  (✅). Outside exponents: needs §4.1.
- **Square roots / fractional powers:** `√(exp(−βΔE)) = exp(−βΔE/2)` is just a
  rational exponent (✅). `√(1+exp)` is algebraic-over-the-exp-ring → ❌.
- **Algebraic constant probabilities** (`1/√2`): need algebraic constants; moderate.
- **β / temperature:** factored as the common positive scale; the residual test is
  implicitly **for all β > 0** (distinct exponent vectors stay independent jointly in
  β). A β-dependent rule is not a single-step DB question.
- **Continuous randomness** (Gaussian proposals, continuous displacements): the BFS
  enumerates *finite discrete* bit-trees; a continuous draw has no finite tree. This
  is a fundamental scope boundary — the state space and every draw must be discrete.
  (Engine-level rejection sampling for non-power-of-two ranges is fine: each path is
  finite; only *user-level* "resample until valid" loops are unbounded and hit
  `maxdepth`.)
- **Unbounded loops / geometric-series weights** (`exp/(1−exp)` arising from
  infinite resampling): the all-reject branch never terminates → `maxdepth`
  `CantHandle`. Recommended pattern: *enumerate the valid set, then choose once*
  (one finite draw), which is fully handled.

---

## 5. Tools available for simplifying weights (today)

- **Hash-consing / interning** of thresholds: structurally-equal subexpressions are
  one object, so equality is an `objectid` check.
- **`Val = num/den`** with the denominator as a *multiset of binomials*; `val_add/
  sub/mul` bring two `Val`s to a common denominator via `ms_unionmax`/`ms_diff`, and
  `val_div` appends a binomial.
- **Automatic exact cancellation:** `bs_mul` adds exponent vectors (so
  `exp(−βL)·exp(βL) → 1`), and `bs_add!` deletes zero coefficients (so the VMMC ratio
  cancels with no special-casing).
- **Denominator clearing + monomial grouping** is the canonical normal form: two
  weights are equal iff their cleared numerators match coefficient-by-coefficient.
  This *is* the simplifier — there is no separate algebraic engine.
- **Not present:** polynomial factoring / GCD beyond the binomial multiset
  bookkeeping. The engine will not discover `1−exp(−2βL) = (1−exp(−βL))(1+exp(−βL))`
  to cancel a denominator; this can bloat denominators but never causes a wrong
  verdict (worst case a spurious `CantHandle` on `val_div`).

---

## 6. Expansion roadmap, ranked

**Tier 1 — DONE (`th_max`, `val_div` generalisation):**
1. ✅ **`th_max`** — implemented; switches on the same `a<b` hyperplane as `th_min`,
   picking the complementary branch.
2. ✅ **`val_div` generalised** to any 1–2-term constant-coefficient binomial
   denominator — `1+exp` (Barker/Glauber), `2−exp`, etc., as well as the original
   `1−exp` (VMMC). The factor is stored explicitly in `Val.den`; the residual clears
   it exactly. (Soundness does not even require sign-definiteness — see §4.2.)
3. ⏭️ Algebraic/irrational **constants** in `th_const`: *deferred*. Re-ranked as hard
   — it needs the same number-field core as Tier 3.9, not a quick win.
4. ⏭️ Many-body atom constructors: *deferred* — trivial but unused (no example
   exercises a 3-body coupling); the algebra already supports it.

**Tier 2 — DONE (#5, #7); #6 partially:**
5. ✅ **ℚ[J] coefficient ring** — implemented. Weight coefficients are now
   multivariate polynomials in the atoms (`Poly`); `th_linear(L)` injects the bare
   value `⟨L,J⟩`. The residual test is "every coefficient polynomial ≡ 0", which (a
   polynomial vanishing on an open chamber) is decidable in ℚ. Constant coefficients
   take a fast path, so existing cases are not slowed.
6. ⚠️ **Nested `min`** / condition-bearing `min` operands: *partially done*. `min`/
   `max` operands may now be condition-free polynomial/exp expressions (handled by
   `eval_static`), and the switch is admitted only when it reduces to a single
   hyperplane with constant coefficients. Truly nested `min`-of-`min` (a piecewise
   switch) still `CantHandle`s — full recursive chamber-splitting is future work.
7. ✅ **General constant-coefficient binomial denominators** — subsumed by #2.

**Tier 3 — hard but in principle exact (leaves a pillar):**
8. **CAD** over polynomial conditions → non-linear conditions (`J11²>J22³`) and,
   combined with §4.4, non-linear exponents. Doubly-exponential; replaces the exact
   LP.
9. **Number-field (cyclotomic) arithmetic** → commensurate trigonometric fields. Deep
   refactor of the numeric core; sound.

**Out of reach in principle (not an implementation gap):**
10. Incommensurate transcendental functions (`cos(√2·r)`, `log`, general `√(sum)`):
    no exact finite representation; exponent equality undecidable.
11. `min`/inequalities between **multi-term exp-sums** as exact decisions: brushes the
    Schanuel conjecture; no known exact algorithm.
12. **Continuous** randomness / continuous state: incompatible with exhaustive
    discrete enumeration.

---

## 7. The safety net

Every limit above is **fail-loud**. Unsupported constructs are mostly *unwriteable*
through the API; the few that reach the engine (`÷` by a >2-term or non-constant-
coefficient denominator, a `min`/`max` whose switch is not a hyperplane, a nested
`min` over condition-bearing operands, a non-integer/non-covariant coordinate, an
arithmetic overflow, a path past `maxdepth`) raise `CantHandle` and abort. The
checker therefore **never trades correctness for coverage**: it either proves the
verdict within the exact class above, or it refuses — it does not approximate.
