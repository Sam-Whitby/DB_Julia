# ============================================================================
# dbc.jl  —  DB_Julia: a Julia-native detailed-balance checker for lattice MCMC
# ============================================================================
# Given a lattice MCMC step (translated to Julia using the random primitives
# defined here), run the ENTIRE verification in Julia:
#
#     state enumeration
#       -> tau-augmented BFS  (translational-invariance check + exact symbolic
#          enumeration of every random-number path)
#       -> ergodicity reachability
#       -> detailed balance  (HiGHS LP chamber enumeration + exact rational
#          grouping with denominators cleared).
#
# DESIGN PRINCIPLE: correctness over speed. The checker must NEVER return a
# false PASS (detailed balance reported as satisfied when it is not) or a false
# FAIL. Anything the engine cannot represent EXACTLY is raised as a `CantHandle`
# error and aborts the run loudly, rather than producing a possibly-wrong
# verdict. All weight arithmetic is exact (Rational{BigInt}).
#
# See README.md for the supported-primitive list, the soundness argument, and
# the documented edge cases that trigger a hard error.
# ============================================================================

using Printf

# ----------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------
struct OutOfBitsException  <: Exception end          # BFS path needs more bits
struct OutOfRangeException <: Exception end          # rejected bit pattern (rejection sampling)
struct CantHandle          <: Exception; msg::String end   # unsupported -> hard abort

cant(msg) = throw(CantHandle(msg))

# ============================================================================
# SECTION 1 — TauNum: exact linear tau-tracking with nonlinear taint
# ============================================================================
# Represents a scalar of the form
#
#     v + cr*tau_r + cc*tau_c        (+ tainted higher-order tau terms)
#
# where v, cr, cc are exact rationals and `tainted` records that a genuinely
# nonlinear tau term (tau_r^2, tau_r*tau_c, ...) was produced and dropped from
# the linear representation. A value is translation-invariant ("tau-free") iff
#
#     cr == 0 && cc == 0 && !tainted
#
# The tau symbols are a pure detection device: the real algorithm runs at
# tau = 0. After detection, `tau0(x)` extracts the genuine value v (the value
# the real lattice algorithm would compute).
#
# Subtraction of two positions carrying the SAME offset cancels tau exactly
# (cr_a - cr_b = 0), which is how pairwise differences become tau-free.
# Squaring a tau-augmented coordinate (cr != 0) produces a tau_r^2 term and
# sets `tainted`, which is how absolute-position energies (quadratic_field) are
# detected as translation-NON-invariant.
# ----------------------------------------------------------------------------

const Q = Rational{BigInt}

struct TauNum
    v::Q
    cr::Q
    cc::Q
    tainted::Bool
end

TauNum(v::Integer)            = TauNum(Q(v), Q(0), Q(0), false)
TauNum(v::Rational)           = TauNum(Q(v), Q(0), Q(0), false)
tau_const(v)                  = TauNum(Q(v), Q(0), Q(0), false)
# tau-augment a coordinate: value r, unit offset along the chosen axis.
tau_r_aug(r::Integer) = TauNum(Q(r), Q(1), Q(0), false)
tau_c_aug(r::Integer) = TauNum(Q(r), Q(0), Q(1), false)

is_tau_free(x::TauNum) = x.cr == 0 && x.cc == 0 && !x.tainted
tau0(x::TauNum)        = x.v          # value at tau = 0 (the real value)

Base.convert(::Type{TauNum}, v::Integer)  = TauNum(v)
Base.convert(::Type{TauNum}, v::Rational) = TauNum(v)
Base.promote_rule(::Type{TauNum}, ::Type{<:Integer})  = TauNum
Base.promote_rule(::Type{TauNum}, ::Type{<:Rational}) = TauNum

Base.:+(a::TauNum, b::TauNum) = TauNum(a.v+b.v, a.cr+b.cr, a.cc+b.cc, a.tainted|b.tainted)
Base.:-(a::TauNum, b::TauNum) = TauNum(a.v-b.v, a.cr-b.cr, a.cc-b.cc, a.tainted|b.tainted)
Base.:-(a::TauNum)            = TauNum(-a.v, -a.cr, -a.cc, a.tainted)

function Base.:*(a::TauNum, b::TauNum)
    # Linear part of the product; flag any nonlinear tau cross-term as taint.
    v  = a.v*b.v
    cr = a.v*b.cr + a.cr*b.v
    cc = a.v*b.cc + a.cc*b.v
    nonlinear = a.tainted | b.tainted |
                (a.cr != 0 && b.cr != 0) |   # tau_r^2
                (a.cc != 0 && b.cc != 0) |   # tau_c^2
                (a.cr != 0 && b.cc != 0) |   # tau_r*tau_c
                (a.cc != 0 && b.cr != 0)
    TauNum(v, cr, cc, nonlinear)
end

Base.:+(a::TauNum, b::Union{Integer,Rational}) = a + TauNum(b)
Base.:+(a::Union{Integer,Rational}, b::TauNum) = TauNum(a) + b
Base.:-(a::TauNum, b::Union{Integer,Rational}) = a - TauNum(b)
Base.:-(a::Union{Integer,Rational}, b::TauNum) = TauNum(a) - b
Base.:*(a::TauNum, b::Union{Integer,Rational}) = a * TauNum(b)
Base.:*(a::Union{Integer,Rational}, b::TauNum) = TauNum(a) * b

function Base.:^(a::TauNum, n::Integer)
    n < 0 && cant("TauNum raised to a negative power")
    r = TauNum(1)
    for _ in 1:n; r = r * a; end
    r
end

# Equality is exact and tau-aware (used only for canonical dedup of tau-free
# values; comparisons that BRANCH go through require_tau_free first).
Base.:(==)(a::TauNum, b::TauNum) = a.v==b.v && a.cr==b.cr && a.cc==b.cc && a.tainted==b.tainted
Base.hash(a::TauNum, h::UInt) = hash((a.v,a.cr,a.cc,a.tainted), h)

# ============================================================================
# SECTION 2 — Atoms, linear forms, particles, geometry
# ============================================================================

# An Atom is a symbolic parameter the energy / weights are linear in:
#   * a coupling atom  couplingJ[a,b,d2]  (canonicalised so a <= b), or
#   * an "extra" field-like parameter (e.g. fieldH), keyed by a Symbol.
struct Atom
    iscoupling::Bool
    a::Int
    b::Int
    d2::Int
    name::Symbol
end
Jc(a::Int, b::Int, d2::Int) = a <= b ? Atom(true, a, b, d2, :_) : Atom(true, b, a, d2, :_)
Xparam(name::Symbol)        = Atom(false, 0, 0, 0, name)

# Deterministic total order so the global atom list has a stable index.
function Base.isless(x::Atom, y::Atom)
    x.iscoupling != y.iscoupling && return x.iscoupling   # couplings before extras
    if x.iscoupling
        return (x.a, x.b, x.d2) < (y.a, y.b, y.d2)
    else
        return x.name < y.name
    end
end

# A LinForm maps atoms to (possibly tau-dependent) coefficients.
const LinForm  = Dict{Atom, TauNum}      # used while building energies (carries tau)
const RatForm  = Dict{Atom, Q}           # tau=0 form stored in conditions / exponents

function addcoef!(lf::LinForm, a::Atom, c::TauNum)
    nc = get(lf, a, TauNum(0)) + c
    if nc.v == 0 && nc.cr == 0 && nc.cc == 0 && !nc.tainted
        delete!(lf, a)
    else
        lf[a] = nc
    end
    lf
end

function linsub(p::LinForm, q::LinForm)::LinForm     # p - q
    r = copy(p)
    for (a, c) in q; addcoef!(r, a, -c); end
    r
end

any_tau_dep(lf::LinForm) = any(!is_tau_free(c) for c in values(lf))

# Substitute tau = 0 and drop coefficients that vanish at tau = 0.
function tau0_form(lf::LinForm)::RatForm
    r = RatForm()
    for (a, c) in lf
        c.v != 0 && (r[a] = c.v)
    end
    r
end

ratscale(f::RatForm, s::Q)::RatForm = RatForm(a => v*s for (a, v) in f)

# ---- particles / states ----
struct Particle
    r::TauNum
    c::TauNum
    t::Int
end
const PState = Vector{Particle}

# tau-augment a concrete (r,c,type) seed particle along both axes.
aug_particle(r::Int, c::Int, t::Int) = Particle(tau_r_aug(r), tau_c_aug(c), t)

# ---- geometry (all flag tau-violation if used on a tau-dependent value) ----

# Periodic Mod of a coordinate value into 0..n-1, flagging tau dependence.
function pmod(x::TauNum, n::Int)::Int
    is_tau_free(x) || tau_violation!("Mod / branch on an absolute (tau-dependent) position")
    v = tau0(x)
    denominator(v) == 1 || cant("non-integer position coordinate: $v")
    mod(Int(numerator(v)), n)
end

# Squared minimum-image distance between two tau-augmented positions.
# Subtraction cancels tau for genuine pairwise differences; if it does not
# (an absolute-position energy), pmod flags the tau-violation.
function pbc_d2(p::Particle, q::Particle, n::Int)::Int
    dra = pmod(p.r - q.r, n)
    dca = pmod(p.c - q.c, n)
    min(dra, n - dra)^2 + min(dca, n - dca)^2
end

# Are two positions the same site under PBC?  (occupancy test; tau-free diff)
same_site(p::Particle, q::Particle, n::Int) =
    pmod(p.r - q.r, n) == 0 && pmod(p.c - q.c, n) == 0

# ============================================================================
# SECTION 3 — BitSeqRNG and the symbolic random primitives
# ============================================================================
# Mirrors RunWithBitsAT in dbc_core.wl. Each BFS path replays the algorithm
# against a fixed bit string. Selection primitives consume bits and multiply
# the exact rational path coefficient; the Metropolis acceptance records a
# SYMBOLIC factor (the clamped Boltzmann threshold) rather than a float, so the
# downstream DB check is algebraically exact.

# Tau-violation is reported through a process-global flag, reset per BFS run,
# so that the geometry helpers above (which have no rng handle) can raise it.
const _TAU     = Ref(false)
const _TAU_MSG = Ref("")
tau_violation!(msg::String) = (_TAU[] = true; _TAU_MSG[] = msg; nothing)

# ----------------------------------------------------------------------------
# Exact rational functions of exp-monomials (BSum / Val)
# ----------------------------------------------------------------------------
# A threshold's value in a fixed chamber is a sum  c * exp(-beta * (L . J)).  For
# VMMC's frustration test the value is a RATIO of such sums (the ratio cancels
# only after the two acceptance factors are multiplied), so we carry values as
#     Val = num / prod_k (1 - exp(-beta * L_k))
# with `num` a Laurent polynomial in the exp-monomials (BSum) and `den` a
# multiset of binomials (1 - exp(-beta*L_k)).  Denominators are only ever
# products of such binomials, so the representation is closed and the DB check
# clears denominators exactly (the residual is a plain polynomial).
const ExpVec = Vector{Q}                  # exponent L over the global atom list
const BSum   = Dict{ExpVec, Q}            # sum_L coeff * exp(-beta * (L.J))

function bs_add!(s::BSum, L::ExpVec, c::Q)
    c == 0 && return s
    v = get(s, L, Q(0)) + c
    v == 0 ? delete!(s, L) : (s[L] = v); s
end
bs_const(c::Q, nA::Int) = c == 0 ? BSum() : BSum(zeros(Q, nA) => c)
bs_addsum(a::BSum, b::BSum) = (r = copy(a); for (L,c) in b; bs_add!(r,L,c); end; r)
bs_sub(a::BSum, b::BSum)    = (r = copy(a); for (L,c) in b; bs_add!(r,L,-c); end; r)
function bs_mul(a::BSum, b::BSum)
    r = BSum()
    for (La,ca) in a, (Lb,cb) in b; bs_add!(r, La .+ Lb, ca*cb); end
    r
end
bs_shift(a::BSum, E::ExpVec) = BSum((L .+ E) => c for (L,c) in a)

struct Val
    num::BSum
    den::Vector{ExpVec}       # multiset of L_k ; den = prod (1 - exp(-beta*L_k))
end
val_const(c::Q, nA::Int) = Val(bs_const(c, nA), ExpVec[])
val_boltz(L::ExpVec)     = Val(BSum(copy(L) => Q(1)), ExpVec[])    # exp(-beta*L)
val_mul(a::Val, b::Val)  = Val(bs_mul(a.num, b.num), vcat(a.den, b.den))

function expand_binoms(dens::Vector{ExpVec}, nA::Int)
    acc = bs_const(Q(1), nA)
    for L in dens
        acc = bs_mul(acc, BSum(zeros(Q,nA) => Q(1), copy(L) => Q(-1)))
    end
    acc
end
val_oneminus(v::Val, nA::Int) = Val(bs_sub(expand_binoms(v.den, nA), v.num), copy(v.den))

function ms_unionmax(a::Vector{ExpVec}, b::Vector{ExpVec})
    cnt = Dict{ExpVec,Int}()
    for L in a; cnt[L] = max(get(cnt,L,0), count(==(L), a)); end
    for L in b; cnt[L] = max(get(cnt,L,0), count(==(L), b)); end
    out = ExpVec[]; for (L,k) in cnt, _ in 1:k; push!(out,L); end; out
end
function ms_diff(big::Vector{ExpVec}, small::Vector{ExpVec})
    rem = copy(big)
    for L in small
        i = findfirst(==(L), rem); i === nothing && cant("denominator multiset diff failed"); deleteat!(rem, i)
    end
    rem
end

# ----------------------------------------------------------------------------
# Conditions and symbolic thresholds (ThExpr)
# ----------------------------------------------------------------------------
# A condition is a linear inequality on the couplings:  lhs (op) 0,  op being
# `<` (strict) or `<=` (non-strict).  It is TRUE in chambers where that holds.
struct Cond
    lhs::RatForm
    strict::Bool        # true: lhs < 0 ; false: lhs <= 0
end
# Canonical hashable key for a RatForm (sorted (atom,coeff) pairs).
_form_key(f::RatForm) = Tuple(sort([(a, v) for (a, v) in f]))
cond_key(c::Cond) = (_form_key(c.lhs), c.strict)

# Subtraction of two energy forms (RatForm).
function rf_sub(a::RatForm, b::RatForm)::RatForm
    r = copy(a); for (k,v) in b; nv = get(r,k,Q(0))-v; nv==0 ? delete!(r,k) : (r[k]=nv); end; r
end
# Flag a tau-violation if an energy form is tau-dependent, then drop tau.
tau0_checked(lf::LinForm)::RatForm =
    (any_tau_dep(lf) && tau_violation!("a threshold/condition depends on an absolute (tau-dependent) position"); tau0_form(lf))

abstract type ThExpr end
struct ThConst <: ThExpr; c::Q; end
struct ThBoltz <: ThExpr; L::RatForm; end                 # exp(-beta * L)
struct ThOp    <: ThExpr; op::Symbol; a::ThExpr; b::ThExpr; end   # :+,:-,:*,:/
struct ThMin   <: ThExpr; a::ThExpr; b::ThExpr; end
struct ThPiece <: ThExpr; clauses::Vector{Tuple{Vector{Cond},ThExpr}}; default::ThExpr; end

# Hash-consing (interning): structurally-equal thresholds share ONE object, so
# the millions of thresholds built during the BFS collapse to a few thousand
# canonical nodes. This makes weight-deduplication an objectid comparison
# instead of a deep structural hash (the dominant cost otherwise). Keys are
# built from already-interned children by objectid, so they stay small.
const _TH_CACHE = Dict{Any,ThExpr}()
reset_th_cache!() = empty!(_TH_CACHE)
_intern(make::Function, key) = get!(make, _TH_CACHE, key)   # `_intern(key) do ... end`

# Translation-facing builders (energies are LinForm so tau is tracked).
th_const(x)                 = _intern((:c, Q(x))) do; ThConst(Q(x)) end
th_boltz(L::LinForm)        = (rl = tau0_checked(L); _intern((:b, _form_key(rl))) do; ThBoltz(rl) end)
th_add(a::ThExpr,b::ThExpr) = _intern((:op,:+,objectid(a),objectid(b))) do; ThOp(:+,a,b) end
th_sub(a::ThExpr,b::ThExpr) = _intern((:op,:-,objectid(a),objectid(b))) do; ThOp(:-,a,b) end
th_mul(a::ThExpr,b::ThExpr) = _intern((:op,:*,objectid(a),objectid(b))) do; ThOp(:*,a,b) end
th_div(a::ThExpr,b::ThExpr) = _intern((:op,:/,objectid(a),objectid(b))) do; ThOp(:/,a,b) end
th_min(a::ThExpr,b::ThExpr) = _intern((:min,objectid(a),objectid(b))) do; ThMin(a,b) end
c_lt(a::LinForm,b::LinForm) = Cond(tau0_checked(linsub(a,b)), true)   # a < b
c_le(a::LinForm)            = Cond(tau0_checked(a), false)            # a <= 0

# A condition on an identically-zero form is constant:  0<0 is always False,
# 0<=0 is always True.  Such guards are resolved at construction so degenerate
# clauses (e.g. a VMMC link where the two distances coincide, making eInit==eFwd
# and the ratio denominator structurally zero) are pruned before evaluation.
is_const_false(c::Cond) = isempty(c.lhs) &&  c.strict
is_const_true(c::Cond)  = isempty(c.lhs) && !c.strict

# Build a Piecewise threshold, pruning clauses whose guard can never hold and
# dropping always-true guards.  A clause whose guard becomes empty always fires,
# so later clauses are unreachable.
function th_piece(clauses, default)
    kept = Tuple{Vector{Cond},ThExpr}[]
    for (guards, val) in clauses
        any(is_const_false, guards) && continue
        g2 = Cond[g for g in guards if !is_const_true(g)]
        push!(kept, (g2, val))
        isempty(g2) && break                            # always fires; rest dead
    end
    key = (:pw, Tuple((Tuple(cond_key(g) for g in gs), objectid(v)) for (gs,v) in kept), objectid(default))
    _intern(key) do; ThPiece(kept, default) end
end

to_vec(rf::RatForm, aidx::Dict{Atom,Int}, nA::Int) =
    (v = zeros(Q, nA); for (a,c) in rf; v[aidx[a]] = c; end; v)

struct ThFactor
    thr::ThExpr
    accepted::Bool
end

mutable struct BitSeqRNG
    bits::Vector{Int}
    pos::Int
    coeff::Q                    # rational selection-probability product
    factors::Vector{ThFactor}   # symbolic acceptance factors, in order
end
BitSeqRNG(bits::Vector{Int}) = BitSeqRNG(bits, 0, Q(1), ThFactor[])

function read_bit!(rng::BitSeqRNG)::Int
    rng.pos += 1
    rng.pos > length(rng.bits) && throw(OutOfBitsException())
    rng.coeff *= 1 // 2
    rng.bits[rng.pos]
end
function read_bits_int!(rng::BitSeqRNG, k::Int)::Int
    acc = 0; for _ in 1:k; acc = acc*2 + read_bit!(rng); end; acc
end

# Bits needed to index 0..n-1, matching Mathematica IntegerLength[n-1, 2].
nbits(n::Int) = n <= 1 ? 0 : ndigits(n - 1; base = 2)

# Uniform choice of an index 1..n with exact rejection-sampling weight 1/n.
function rand_choice_index!(rng::BitSeqRNG, n::Int)::Int
    n == 0 && cant("RandomChoice over an empty list")
    n == 1 && return 1
    k   = nbits(n)
    val = read_bits_int!(rng, k)
    val >= n && throw(OutOfRangeException())
    rng.coeff *= Q(2)^k // n
    val + 1
end
rand_choice!(rng::BitSeqRNG, list::AbstractVector) = list[rand_choice_index!(rng, length(list))]

# RandomInteger[{lo, hi}] with exact rejection sampling.
function rand_integer!(rng::BitSeqRNG, lo::Int, hi::Int)::Int
    n = hi - lo + 1
    n <= 0 && cant("RandomInteger[{$lo,$hi}]: inverted range (hi < lo) — likely an algorithm bug")
    n == 1 && return lo
    k   = nbits(n)
    val = read_bits_int!(rng, k)
    val >= n && throw(OutOfRangeException())
    rng.coeff *= Q(2)^k // n
    lo + val
end

# General acceptance test: mirrors  RandomReal[] < thr  for an arbitrary symbolic
# threshold.  Always reads one bit (always-read policy); the symbolic factor is
# recorded (thr on accept, 1-thr on reject) for exact downstream evaluation.
function accept!(rng::BitSeqRNG, thr::ThExpr)::Bool
    rng.pos += 1
    rng.pos > length(rng.bits) && throw(OutOfBitsException())
    accepted = rng.bits[rng.pos] == 1
    push!(rng.factors, ThFactor(thr, accepted))
    accepted
end

# Metropolis acceptance: RandomReal[] < Piecewise[{{1, dE<=0}}, exp(-beta*exponent)].
function metropolis!(rng::BitSeqRNG, dE::LinForm; exponent::LinForm = dE)::Bool
    (any_tau_dep(dE) || any_tau_dep(exponent)) &&
        tau_violation!("acceptance threshold depends on an absolute (tau-dependent) position")
    # dE identically 0  =>  threshold 1  =>  always accept, no bit.
    isempty(tau0_form(dE)) && return true
    accept!(rng, th_piece([([c_le(dE)], th_const(1))], th_boltz(exponent)))
end

# ============================================================================
# SECTION 4 — BFS engine (per-state exhaustive path enumeration)
# ============================================================================
# A concrete state is a sorted vector of (row, col, type) integer triples.
const CState = Vector{NTuple{3,Int}}

intval(x::TauNum)::Int = (denominator(tau0(x)) == 1 ?
    Int(numerator(tau0(x))) : cant("non-integer position coordinate: $(tau0(x))"))

# Substitute tau = 0, apply PBC into 1..n, and sort -> canonical concrete state.
# tau=0 on a next-state position is always valid: a translation-invariant
# algorithm's next state is translation-COVARIANT (it shifts with the lattice),
# which is expected and must NOT be flagged.
function norm_state(s::PState, n::Int)::CState
    cs = NTuple{3,Int}[(mod(intval(p.r) - 1, n) + 1, mod(intval(p.c) - 1, n) + 1, p.t)
                       for p in s]
    sort!(cs)
    cs
end

struct Leaf
    next::CState
    coeff::Q
    factors::Vector{ThFactor}
end

# Exhaustive BFS over all bit strings for one (tau-augmented) seed state.
# Returns the leaves; raises CantHandle if any path exceeds maxdepth (an
# incomplete tree would silently drop transitions and is treated as fatal).
function build_state_leaves(algo, seed::PState, n::Int, maxdepth::Int)::Vector{Leaf}
    leaves = Leaf[]
    queue  = Vector{Int}[Int[]]
    while !isempty(queue)
        bits = popfirst!(queue)
        rng  = BitSeqRNG(bits)
        try
            nxt = algo(rng, seed)::PState
            push!(leaves, Leaf(norm_state(nxt, n), rng.coeff, copy(rng.factors)))
        catch e
            if e isa OutOfBitsException
                if length(bits) < maxdepth
                    push!(queue, [bits; 0]); push!(queue, [bits; 1])
                else
                    cant("BFS incomplete: a random-number path reached maxdepth=$maxdepth " *
                         "before the algorithm returned a state; increase maxdepth")
                end
            elseif e isa OutOfRangeException
                # rejected (out-of-range) bit pattern in rejection sampling — drop
            else
                rethrow(e)
            end
        end
    end
    leaves
end

# ---- concrete float evaluation of a leaf weight (sanity testing only) ----
_rfval(rf::RatForm, J::Dict{Atom,Float64}) = sum(Float64(v)*get(J,a,0.0) for (a,v) in rf; init=0.0)
_guard_true(c::Cond, J) = c.strict ? _rfval(c.lhs,J) < 0 : _rfval(c.lhs,J) <= 0
function eval_th_float(t::ThExpr, J::Dict{Atom,Float64}, beta::Float64)::Float64
    if t isa ThConst; Float64(t.c)
    elseif t isa ThBoltz; exp(-beta * _rfval(t.L, J))
    elseif t isa ThOp
        a = eval_th_float(t.a,J,beta); b = eval_th_float(t.b,J,beta)
        t.op === :+ ? a+b : t.op === :- ? a-b : t.op === :* ? a*b : a/b
    elseif t isa ThMin; min(eval_th_float(t.a,J,beta), eval_th_float(t.b,J,beta))
    else  # ThPiece
        for (guards,val) in t.clauses
            all(_guard_true(g,J) for g in guards) && return eval_th_float(val,J,beta)
        end
        eval_th_float(t.default,J,beta)
    end
end
function eval_leaf(lf::Leaf, J::Dict{Atom,Float64}, beta::Float64)::Float64
    w = Float64(lf.coeff)
    for f in lf.factors
        th = eval_th_float(f.thr, J, beta)
        w *= f.accepted ? th : (1.0 - th)
    end
    w
end

# ============================================================================
# SECTION 5 — State enumeration
# ============================================================================
# Ordered selections (k-permutations without repetition) of `items`.
function _kperms(items::Vector{T}, k::Int) where {T}
    out = Vector{T}[]
    n = length(items)
    used = falses(n)
    cur = Vector{T}(undef, k)
    function rec(d)
        if d > k
            push!(out, copy(cur)); return
        end
        for i in 1:n
            used[i] && continue
            used[i] = true; cur[d] = items[i]
            rec(d + 1)
            used[i] = false
        end
    end
    rec(1)
    out
end

# All distinct N-particle states for the given type multiset on the n x n torus.
function enumerate_states(types::Vector{Int}, n::Int)::Vector{CState}
    st  = sort(types)
    pos = [(r, c) for r in 1:n for c in 1:n]
    seen = Set{CState}()
    out  = CState[]
    for perm in _kperms(pos, length(st))
        cs = sort(NTuple{3,Int}[(perm[k][1], perm[k][2], st[k]) for k in 1:length(st)])
        if !(cs in seen)
            push!(seen, cs); push!(out, cs)
        end
    end
    out
end

# Theoretical combinatorial count: P(S,N) / prod(multiplicity!).
function theoretical_count(types::Vector{Int}, n::Int)
    S = n^2; N = length(types)
    num = prod(BigInt(S - k) for k in 0:N-1)
    den = prod(factorial(BigInt(c)) for c in values(_counts(types)))
    num ÷ den
end
function _counts(xs)
    d = Dict{Int,Int}()
    for x in xs; d[x] = get(d, x, 0) + 1; end
    d
end

# Concrete state -> tau-FREE PState (for energy evaluation of real states).
concrete_pstate(cs::CState)::PState = PState([Particle(TauNum(r), TauNum(c), t) for (r, c, t) in cs])
# Concrete state -> tau-AUGMENTED PState (BFS seed for the tau-check).
augmented_pstate(cs::CState)::PState = PState([aug_particle(r, c, t) for (r, c, t) in cs])

# ============================================================================
# SECTION 6 — Transition build (translation-orbit reduction) + ergodicity
# ============================================================================
# A translation-invariant algorithm has a translation-EQUIVARIANT transition
# matrix, so we BFS only one representative per translation orbit (~nGrid^2 fewer
# BFS runs) and obtain every other state's transitions by translating the rep's
# leaves. The leaf WEIGHTS are translation-invariant (they depend on types and
# pairwise distances, not absolute positions), so all states in an orbit share
# the same symbolic weights — the unique-weight set comes from the reps alone.
#
# This reduction is valid ONLY when translation invariance holds. If the tau-BFS
# from a rep flags a violation, we fall back to a direct BFS from EVERY state
# (no equivariance assumed), which is correct for any algorithm.

# Translate a concrete state by (dr,dc) on the n-torus.
translate_cstate(cs::CState, dr::Int, dc::Int, n::Int)::CState =
    sort(NTuple{3,Int}[(mod(r-1+dr,n)+1, mod(c-1+dc,n)+1, t) for (r,c,t) in cs])

# Partition states into translation orbits. Returns the reps and, per state,
# (rep, (dr,dc)) such that translate(rep, dr,dc) == state.
function translation_orbits(states::Vector{CState}, n::Int)
    repof = Dict{CState,CState}(); gof = Dict{CState,Tuple{Int,Int}}(); reps = CState[]
    for cs in states
        haskey(repof, cs) && continue
        orbit = Dict{CState,Tuple{Int,Int}}()
        for dr in 0:n-1, dc in 0:n-1
            t = translate_cstate(cs, dr, dc, n)
            haskey(orbit, t) || (orbit[t] = (dr, dc))
        end
        rep = minimum(keys(orbit)); grep = orbit[rep]; push!(reps, rep)
        for (t, gt) in orbit
            repof[t] = rep
            gof[t] = (mod(gt[1]-grep[1], n), mod(gt[2]-grep[2], n))  # translate(rep,g)=t
        end
    end
    reps, repof, gof
end

struct BFSResult
    states   :: Vector{CState}
    idx      :: Dict{CState,Int}
    uweights :: Vector{Leaf}                 # unique (representative) leaf weights
    trans    :: Vector{Tuple{Int,Int,Int}}   # (src, dst, weight-index), src != dst
    tau_free :: Bool
    tau_msg  :: String
end

function build_transitions(algo, energy, states::Vector{CState}, n::Int, maxdepth::Int)::BFSResult
    reset_th_cache!()                       # fresh interning table for this run
    idx = Dict(cs => i for (i, cs) in enumerate(states))
    uweights = Leaf[]; uw_idx = Dict{Any,Int}()
    widx!(lf::Leaf) = get!(uw_idx, _weight_key(lf)) do; push!(uweights, lf); length(uweights) end
    trans = Tuple{Int,Int,Int}[]
    tau_free = true; tau_msg = ""

    reps, repof, gof = translation_orbits(states, n)
    rep_leaves = Dict{CState,Vector{Leaf}}()
    for rep in reps
        _TAU[] = false; _TAU_MSG[] = ""
        rep_leaves[rep] = build_state_leaves(algo, augmented_pstate(rep), n, maxdepth)
        if _TAU[]; tau_free = false; isempty(tau_msg) && (tau_msg = _TAU_MSG[]); end
    end

    if tau_free
        # Dedup the rep leaves' weights ONCE (by object identity in the orbit
        # expansion below), then translate each rep leaf to every orbit member.
        wi_of = IdDict{Leaf,Int}()
        for rep in reps, lf in rep_leaves[rep]; wi_of[lf] = widx!(lf); end
        for (s, si) in idx
            rep = repof[s]; (dr, dc) = gof[s]
            for lf in rep_leaves[rep]
                dst = idx[translate_cstate(lf.next, dr, dc, n)]
                dst != si && push!(trans, (si, dst, wi_of[lf]))
            end
        end
    else
        # Fallback: direct BFS from every state (no equivariance assumed).
        for (s, si) in idx
            lvs = haskey(rep_leaves, s) ? rep_leaves[s] :
                  build_state_leaves(algo, augmented_pstate(s), n, maxdepth)
            for lf in lvs
                wi = widx!(lf); dst = idx[lf.next]
                dst != si && push!(trans, (si, dst, wi))
            end
        end
    end
    BFSResult(states, idx, uweights, trans, tau_free, tau_msg)
end

# Reachability ergodicity: BFS over the directed transition graph from the seed.
function check_ergodicity(bfs::BFSResult, seed::CState)
    nS = length(bfs.states)
    adj = [Set{Int}() for _ in 1:nS]
    for (s, d, _) in bfs.trans; push!(adj[s], d); end
    seedi = bfs.idx[seed]
    seen = Set{Int}([seedi]); queue = [seedi]
    while !isempty(queue)
        u = popfirst!(queue)
        for v in adj[u]
            if !(v in seen); push!(seen, v); push!(queue, v); end
        end
    end
    (ergodic = length(seen) == nS, reached = length(seen), total = nS)
end

# ============================================================================
# SECTION 7 — Detailed-balance check (conditions, chambers, exact residual)
# ============================================================================
# The leaf weights are products of symbolic acceptance factors (ThExpr). The
# branch CONDITIONS (linear inequalities on the couplings) carve coupling space
# into chambers; within each chamber every factor resolves to an exact rational
# function of exp-monomials (a Val). For each communicating pair the detailed-
# balance residual is formed and its denominators cleared, leaving a polynomial
# whose coefficients must all vanish — checked exactly with Rational{BigInt}.

using HiGHS

# Unique-weight key: thresholds are interned (hash-consed), so structurally
# equal thresholds are the SAME object and objectid identifies them in O(1).
_weight_key(lf::Leaf) = (lf.coeff, Tuple((objectid(f.thr), f.accepted) for f in lf.factors))

vecof(f::RatForm, aidx::Dict{Atom,Int}, nA::Int) =
    (v = zeros(Q, nA); for (a, c) in f; v[aidx[a]] = c; end; v)

# ---- atom collection: every coupling that appears anywhere ----
function _collect_atoms_th!(set::Set{Atom}, t::ThExpr)
    if t isa ThConst
    elseif t isa ThBoltz; for a in keys(t.L); push!(set,a); end
    elseif t isa ThOp; _collect_atoms_th!(set,t.a); _collect_atoms_th!(set,t.b)
    elseif t isa ThMin; _collect_atoms_th!(set,t.a); _collect_atoms_th!(set,t.b)
    else; for (gs,v) in t.clauses; for g in gs, a in keys(g.lhs); push!(set,a); end; _collect_atoms_th!(set,v); end; _collect_atoms_th!(set,t.default)
    end
end

# ---- "static" Val evaluation of a condition-free ThExpr (ThMin operands) ----
# Errors if it meets a ThPiece/ThMin, i.e. the operands of a Min must not
# themselves contain conditions (true for VMMC; fail-loud otherwise).
function eval_static(t::ThExpr, aidx, nA)::Val
    if t isa ThConst; val_const(t.c, nA)
    elseif t isa ThBoltz; val_boltz(to_vec(t.L, aidx, nA))
    elseif t isa ThOp
        a = eval_static(t.a,aidx,nA); b = eval_static(t.b,aidx,nA)
        t.op === :+ ? val_add(a,b,nA) : t.op === :- ? val_sub(a,b,nA) :
        t.op === :* ? val_mul(a,b)    : val_div(a,b)
    else
        cant("Min over a condition-bearing expression is not supported")
    end
end

# Condition implied by Min[a,b] (TRUE iff a < b), as (eff_lhs, strict).
# Requires the cleared-denominator difference a-b to be a balanced binomial
# c*(exp(-bL1) - exp(-bL2)); then a<b reduces to the linear form (L1-L2)>0.
function min_condition(a::ThExpr, b::ThExpr, aidx, nA)
    va = eval_static(a, aidx, nA); vb = eval_static(b, aidx, nA)
    numer = bs_sub(bs_mul(va.num, expand_binoms(vb.den, nA)),
                   bs_mul(vb.num, expand_binoms(va.den, nA)))   # numerator of (a-b)
    isempty(numer) && return nothing                            # a == b, no condition
    length(numer) == 2 || cant("Min condition is not a linear hyperplane (numerator has $(length(numer)) terms)")
    (k1,c1),(k2,c2) = collect(numer)
    c1 == -c2 || cant("Min condition numerator is an unbalanced binomial (not a hyperplane)")
    kp, km = c1 > 0 ? (k1,k2) : (k2,k1)                         # +coeff key, -coeff key
    # a<b  <=>  numer<0  <=>  exp(-b*kp) < exp(-b*km)  <=>  kp>km  <=>  (km-kp)<0
    eff = kp .- km                                              # sigma=1 (a<b) iff eff.J > 0
    (eff, true)
end

# ---- exact Val evaluation of a ThExpr under a chamber sign pattern ----
function eval_val(t::ThExpr, σ::Vector{Int}, ctx, aidx, nA)::Val
    if t isa ThConst; val_const(t.c, nA)
    elseif t isa ThBoltz; val_boltz(to_vec(t.L, aidx, nA))
    elseif t isa ThOp
        a = eval_val(t.a,σ,ctx,aidx,nA); b = eval_val(t.b,σ,ctx,aidx,nA)
        t.op === :+ ? val_add(a,b,nA) : t.op === :- ? val_sub(a,b,nA) :
        t.op === :* ? val_mul(a,b)    : val_div(a,b)
    elseif t isa ThMin
        idx = ctx.thmin[objectid(t)]
        idx == 0 ? eval_val(t.a,σ,ctx,aidx,nA) :                 # a==b
            (σ[idx]==1 ? eval_val(t.a,σ,ctx,aidx,nA) : eval_val(t.b,σ,ctx,aidx,nA))
    else  # ThPiece
        for (gs,v) in t.clauses
            all(σ[ctx.cidx[(-vecof(g.lhs,aidx,nA), g.strict)]]==1 for g in gs) &&
                return eval_val(v,σ,ctx,aidx,nA)
        end
        eval_val(t.default,σ,ctx,aidx,nA)
    end
end

val_add(a::Val,b::Val,nA) = (D=ms_unionmax(a.den,b.den);
    Val(bs_addsum(bs_mul(a.num,expand_binoms(ms_diff(D,a.den),nA)),
                  bs_mul(b.num,expand_binoms(ms_diff(D,b.den),nA))), D))
val_sub(a::Val,b::Val,nA) = (D=ms_unionmax(a.den,b.den);
    Val(bs_sub(bs_mul(a.num,expand_binoms(ms_diff(D,a.den),nA)),
               bs_mul(b.num,expand_binoms(ms_diff(D,b.den),nA))), D))
# a / b  where b must be a single binomial (1 - exp(-beta*L)).
function val_div(a::Val, b::Val)
    isempty(b.den) && length(b.num)==2 || cant("division by a non-binomial threshold")
    z = nothing; L = nothing
    for (k,c) in b.num
        if all(==(0), k); c==1 || cant("bad binomial numerator"); z=k
        else; c==-1 || cant("bad binomial numerator"); L=k; end
    end
    (z===nothing || L===nothing) && cant("division denominator is not (1 - exp)")
    Val(a.num, vcat(a.den, [L]))
end

# ---- the DB model ----
# Leaf weights stay symbolic (representative ThExpr leaves); their per-chamber
# Val is evaluated lazily and cached, and each pair is checked once per DISTINCT
# projection of the chambers onto that pair's active conditions (usually a
# handful), instead of once per chamber.
struct DBModel
    aidx           :: Dict{Atom,Int}
    nA             :: Int
    cond_eff_lhs   :: Vector{Vector{Q}}        # sigma=1 iff eff_lhs . J >= 0 (>0 if strict)
    cond_is_strict :: Vector{Bool}
    energy_coeffs  :: Vector{Vector{Q}}
    ctx                                        # (cidx, thmin) for ThExpr evaluation
    uweights       :: Vector{Leaf}             # representative leaf per unique weight
    uw_active      :: Vector{Vector{Int}}      # active cond indices per unique weight
    pairs          :: Vector{Tuple{Int,Int}}
    ij_srcs        :: Vector{Vector{Int}}
    ji_srcs        :: Vector{Vector{Int}}
end

function build_dbmodel(bfs::BFSResult, energy)::DBModel
    uweights = bfs.uweights                                  # already deduped (rep weights)

    # --- atoms (couplings) over the unique thresholds and the state energies ---
    state_energy = [tau0_form(energy(concrete_pstate(cs))) for cs in bfs.states]
    atomset = Set{Atom}()
    for lf in uweights, f in lf.factors; _collect_atoms_th!(atomset, f.thr); end
    for se in state_energy, a in keys(se); push!(atomset, a); end
    atoms = sort(collect(atomset)); aidx = Dict(a=>i for (i,a) in enumerate(atoms)); nA = length(atoms)
    energy_coeffs = [vecof(se, aidx, nA) for se in state_energy]

    # --- condition registry: (eff_lhs, strict) -> index ; ThMin object -> index ---
    cidx = Dict{Tuple{Vector{Q},Bool},Int}()
    eff_list = Vector{Q}[]; strict_list = Bool[]
    register!(eff::Vector{Q}, strict::Bool) = get!(cidx, (eff,strict)) do
        push!(eff_list, eff); push!(strict_list, strict); length(eff_list)
    end
    thmin = Dict{UInt,Int}()                       # ThMin objectid -> cond index (0 = a==b)
    function scan!(t::ThExpr)
        if t isa ThOp; scan!(t.a); scan!(t.b)
        elseif t isa ThMin
            if !haskey(thmin, objectid(t))     # shared interned node: derive once
                mc = min_condition(t.a, t.b, aidx, nA)
                thmin[objectid(t)] = mc === nothing ? 0 : register!(-mc[1], mc[2])  # sigma=1 iff a<b
            end
            scan!(t.a); scan!(t.b)
        elseif t isa ThPiece
            for (gs,v) in t.clauses
                for g in gs; register!(-vecof(g.lhs,aidx,nA), g.strict); end
                scan!(v)
            end
            scan!(t.default)
        end
    end
    for lf in uweights, f in lf.factors; scan!(f.thr); end
    ctx = (cidx=cidx, thmin=thmin)

    # --- transitions -> per-pair weight-index lists ---
    trans = Dict{Tuple{Int,Int},Vector{Int}}()
    for (s, d, wi) in bfs.trans; push!(get!(trans,(s,d),Int[]), wi); end

    # --- per unique weight: active conds ---
    function factor_conds(t::ThExpr, acc::Set{Int})
        if t isa ThOp; factor_conds(t.a,acc); factor_conds(t.b,acc)
        elseif t isa ThMin; (i=thmin[objectid(t)]; i!=0 && push!(acc,i)); factor_conds(t.a,acc); factor_conds(t.b,acc)
        elseif t isa ThPiece
            for (gs,v) in t.clauses; for g in gs; push!(acc, cidx[(-vecof(g.lhs,aidx,nA),g.strict)]); end; factor_conds(v,acc); end
            factor_conds(t.default,acc)
        end
    end
    uw_active = Vector{Vector{Int}}(undef, length(uweights))
    for (wi, lf) in enumerate(uweights)
        accset = Set{Int}(); for f in lf.factors; factor_conds(f.thr, accset); end
        uw_active[wi] = sort(collect(accset))
    end

    pairset = Set{Tuple{Int,Int}}(); for (i,j) in keys(trans); push!(pairset,(min(i,j),max(i,j))); end
    pairs = sort(collect(pairset))
    ij_srcs = [get(trans,(a,b),Int[]) for (a,b) in pairs]
    ji_srcs = [get(trans,(b,a),Int[]) for (a,b) in pairs]

    DBModel(aidx, nA, eff_list, strict_list, energy_coeffs, ctx,
            uweights, uw_active, pairs, ij_srcs, ji_srcs)
end

# ---- Phase 2: chamber enumeration via HiGHS LP BFS + degenerate filter ----
function _is_feasible(sigma::Vector{Int}, eff::Vector{Vector{Q}},
                     strict::Vector{Bool}, nA::Int, eps::Float64)::Bool
    isempty(sigma) && return true
    h = HiGHS.Highs_create()
    HiGHS.Highs_setBoolOptionValue(h, "output_flag", false)
    for _ in 1:nA
        HiGHS.Highs_addVar(h, -1e10, 1e10)
    end
    inds = Int32[j - 1 for j in 1:nA]
    for i in eachindex(sigma)
        s = sigma[i] == 1 ? 1.0 : -1.0
        coeffs = Float64[s * Float64(eff[i][j]) for j in 1:nA]
        lb = ((sigma[i] == 1) == strict[i]) ? eps : 0.0
        HiGHS.Highs_addRow(h, lb, 1e30, nA, inds, coeffs)
    end
    HiGHS.Highs_run(h)
    st = HiGHS.Highs_getModelStatus(h)
    HiGHS.Highs_destroy(h)
    st == 7   # kOptimal
end

function _filter_degenerate(feasible, eff, strict, k)
    contra_ff = Tuple{Int,Int}[]; contra_tt = Tuple{Int,Int}[]
    for i in 1:k, j in (i+1):k
        if eff[i] == -eff[j]
            if strict[i] && strict[j]
                push!(contra_ff, (i, j))
            elseif !strict[i] && !strict[j]
                push!(contra_tt, (i, j))
            end
        end
    end
    (isempty(contra_ff) && isempty(contra_tt)) && return feasible
    filter(feasible) do s
        !any(s[i] == 0 && s[j] == 0 for (i, j) in contra_ff) &&
        !any(s[i] == 1 && s[j] == 1 for (i, j) in contra_tt)
    end
end

function enumerate_chambers(m::DBModel)::Vector{Vector{Int}}
    k = length(m.cond_eff_lhs); nA = m.nA
    k == 0 && return [Int[]]
    # generic interior witness: J*_a = 100^a guarantees no real condition is 0.
    Jstar = Q[Q(100)^a for a in 1:nA]
    initial = Int[ (sum(m.cond_eff_lhs[i][a] * Jstar[a] for a in 1:nA) >= 0) ? 1 : 0
                   for i in 1:k ]
    eps = 1e-6
    visited = Set{Vector{Int}}([copy(initial)])
    feasible = [copy(initial)]; queue = [copy(initial)]
    while !isempty(queue)
        s = popfirst!(queue)
        for i in 1:k
            s2 = copy(s); s2[i] = 1 - s2[i]
            s2 in visited && continue
            push!(visited, copy(s2))
            if _is_feasible(s2, m.cond_eff_lhs, m.cond_is_strict, nA, eps)
                push!(feasible, copy(s2)); push!(queue, copy(s2))
            end
        end
    end
    _filter_degenerate(feasible, m.cond_eff_lhs, m.cond_is_strict, k)
end

# ---- Phase 3: exact DB check (lazy Val cache + per-pair condition projection) ----
# Returns (pass, violations) where each violation is (i, j, chamber_index).
function run_db_check(m::DBModel)
    chambers = enumerate_chambers(m); nA = m.nA; nC = length(m.cond_eff_lhs)
    # Lazy, cached leaf Val: depends only on the projection of sigma onto the
    # weight's active conditions, so one evaluation serves every chamber sharing
    # that projection.
    cache = Dict{Tuple{Int,Vector{Int}}, Val}()
    function leaf_val(wi::Int, σ::Vector{Int})
        asg = Int[σ[c] for c in m.uw_active[wi]]
        get!(cache, (wi, asg)) do
            v = val_const(m.uweights[wi].coeff, nA)
            for f in m.uweights[wi].factors
                fv = eval_val(f.thr, σ, m.ctx, m.aidx, nA)
                v = val_mul(v, f.accepted ? fv : val_oneminus(fv, nA))
            end
            v
        end
    end
    # Residual numerator for one pair under one chamber (denominators cleared).
    function residual_zero(p::Int, σ::Vector{Int})
        i, j = m.pairs[p]; ei = m.energy_coeffs[i]; ej = m.energy_coeffs[j]
        D = ExpVec[]
        for wi in m.ij_srcs[p]; D = ms_unionmax(D, leaf_val(wi,σ).den); end
        for wi in m.ji_srcs[p]; D = ms_unionmax(D, leaf_val(wi,σ).den); end
        res = BSum()
        for wi in m.ij_srcs[p]
            v = leaf_val(wi,σ)
            for (L,c) in bs_mul(bs_shift(v.num, ei), expand_binoms(ms_diff(D,v.den), nA)); bs_add!(res,L,c); end
        end
        for wi in m.ji_srcs[p]
            v = leaf_val(wi,σ)
            for (L,c) in bs_mul(bs_shift(v.num, ej), expand_binoms(ms_diff(D,v.den), nA)); bs_add!(res,L,-c); end
        end
        isempty(res)
    end

    violations = Tuple{Int,Int,Int}[]
    for p in eachindex(m.pairs)
        (isempty(m.ij_srcs[p]) && isempty(m.ji_srcs[p])) && continue
        # Active conditions for this pair: only these distinguish its chambers.
        apset = Set{Int}()
        for wi in m.ij_srcs[p]; union!(apset, m.uw_active[wi]); end
        for wi in m.ji_srcs[p]; union!(apset, m.uw_active[wi]); end
        ap = sort(collect(apset))
        seen = Set{Vector{Int}}()
        for (ridx, σ) in enumerate(chambers)
            proj = Int[σ[c] for c in ap]
            proj in seen && continue
            push!(seen, proj)
            residual_zero(p, σ) || push!(violations, (m.pairs[p][1], m.pairs[p][2], ridx))
        end
    end
    (isempty(violations), violations, length(chambers))
end
