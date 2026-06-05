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
#       -> detailed balance  (exact-rational-LP chamber enumeration + exact
#          rational grouping with denominators cleared).
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

# Exact rational arithmetic. Int128 (not BigInt) for speed: lattice coordinates,
# distances, selection weights and Boltzmann exponents are tiny, and Int128 gives
# ~38 decimal digits of headroom. Crucially this is SOUND, not a gamble: Julia's
# Rational{Int128} arithmetic is overflow-CHECKED and throws OverflowError on
# overflow (verified), so an out-of-range computation aborts loudly rather than
# silently wrapping to a wrong value. check.jl catches that and reports it as a
# hard error (never a wrong verdict). For the system sizes this tool targets
# (a few particles on lattices up to ~5x5) overflow does not occur.
const Q = Rational{Int128}

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

# ---- rotation/reflection (point-group) taint flag — see SECTION 1c -----------
# `_ROT_PROBE` is true ONLY while the point-group-equivariance probe runs. When on,
# a bare position+offset (TauNum ± Integer/Rational) is a rotation violation: the
# only point-group-COVARIANT way to shift a position is `move(p, d)` with d a
# DECLARED direction (which bypasses this via a direct TauNum constructor). When off
# (the normal / fallback BFS, and every non-rotation example) arithmetic is normal,
# so this adds only one Bool read on the position-add hot path.
const _ROT_PROBE = Ref(false)
const _ROT_BAD   = [false]                                 # thread-local, like _TAU
const _ROT_MSG   = [""]
rot_violation!(m::String) = (i = Threads.threadid(); _ROT_BAD[i] = true; _ROT_MSG[i] = m; nothing)
@inline _rotchk() = (_ROT_PROBE[] && rot_violation!(
    "bare position±offset during the point-group probe; shift via move(p, d) with a declared direction"); nothing)

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

Base.:+(a::TauNum, b::Union{Integer,Rational}) = (_rotchk(); a + TauNum(b))
Base.:+(a::Union{Integer,Rational}, b::TauNum) = (_rotchk(); TauNum(a) + b)
Base.:-(a::TauNum, b::Union{Integer,Rational}) = (_rotchk(); a - TauNum(b))
Base.:-(a::Union{Integer,Rational}, b::TauNum) = (_rotchk(); TauNum(a) - b)
Base.:*(a::TauNum, b::Union{Integer,Rational}) = a * TauNum(b)
Base.:*(a::Union{Integer,Rational}, b::TauNum) = TauNum(a) * b

function Base.:^(a::TauNum, n::Integer)
    n < 0 && cant("TauNum raised to a negative power")
    r = TauNum(1)
    for _ in 1:n; r = r * a; end
    r
end

# Equality on positions is allowed ONLY when the result is translation-safe:
# either both values are tau-free, or they share the SAME tau-coefficients (so
# their difference is tau-free, e.g. comparing two live particle rows). Any other
# equality test mixes an absolute position into a branch and would be a silent
# translation leak, so it is a hard error pushing the user toward same_site/pbc_d2.
# Ordering positions is absolute by nature and is forbidden outright.
function Base.:(==)(a::TauNum, b::TauNum)
    (a.cr == b.cr && a.cc == b.cc && !a.tainted && !b.tainted) && return a.v == b.v
    cant("equality test on an absolute (tau-dependent) position; use same_site / pbc_d2")
end
Base.isless(::TauNum, ::TauNum) =
    cant("ordering comparison on lattice positions is not translation-safe; compare distances via pbc_d2")
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
# Jc is a TRUSTED species primitive: building a coupling atom from two species
# labels is equivariant, so a tagged label is unwrapped WITHOUT flagging (`_u` is
# defined with TypeTag in SECTION 1b). The internal a<=b canonicalisation is atom
# bookkeeping, not a user branch.
Jc(a, b, d2::Int) = Jc(_u(a), _u(b), d2)
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

# ============================================================================
# SECTION 1b — TypeTag: a species-label tag for certifying species-equivariance
# ============================================================================
# A permutation of the species labels relabels the (symbolic) coupling atoms, so
# it is a symmetry of detailed balance when the algorithm is species-EQUIVARIANT.
# To certify that from a SINGLE BFS — so the tau-BFS can be reduced over species
# orbits, not just translation orbits — we run the BFS with each label wrapped in a
# TypeTag that permits only EQUIVARIANT uses (equality between labels, hashing, and
# atom construction via Jc) and FLAGS any use of the ABSOLUTE label (comparison to
# a constant, ordering, arithmetic, coercion to Int). If a whole BFS runs unflagged
# and every output label is itself an (inherited) tag, the move is species-
# equivariant. This is the exact analogue of TauNum/`tau` for the discrete species
# group. The flag is thread-local and deterministic, like the tau flag.
#
# Soundness rests on the same discipline as tau: the raw value `t.v` must not be
# read inside an algorithm (it bypasses the tag). Reads via the documented API
# (`p.t` used in `==`, `Jc`, carried into a new Particle) are safe; coercion to Int
# is intercepted and flagged.
struct TypeTag; v::Int; end
TypeTag(t::TypeTag) = t                                   # idempotent

const _SPECIES_BAD = [false]                              # thread-local, like _TAU
const _SP_MSG      = [""]
species_violation!(m::String) =
    (i = Threads.threadid(); _SPECIES_BAD[i] = true; _SP_MSG[i] = m; nothing)

# ALLOWED (species-equivariant) operations.
Base.:(==)(a::TypeTag, b::TypeTag) = a.v == b.v
Base.hash(a::TypeTag, h::UInt)     = hash(a.v, h)
# FLAGGED operations (they depend on the absolute label, not the equality structure).
Base.:(==)(a::TypeTag, b::Integer) = (species_violation!("species label compared to constant $b"); a.v == b)
Base.:(==)(a::Integer, b::TypeTag) = (species_violation!("constant $a compared to species label"); a == b.v)
Base.isless(a::TypeTag, b::TypeTag)  = (species_violation!("species labels ordered"); a.v < b.v)
Base.isless(a::TypeTag, b::Integer)  = (species_violation!("species label ordered vs constant"); a.v < b)
Base.isless(a::Integer, b::TypeTag)  = (species_violation!("constant ordered vs species label"); a < b.v)
Base.convert(::Type{<:Integer}, t::TypeTag) = (species_violation!("species label coerced to Int"); t.v)
_u(x::TypeTag) = x.v; _u(x) = x
for op in (:+, :-, :*)
    @eval Base.$op(a::TypeTag, b) = (species_violation!("species-label arithmetic ($($(QuoteNode(op))))"); $op(a.v, _u(b)))
    @eval Base.$op(a, b::TypeTag) = (species_violation!("species-label arithmetic ($($(QuoteNode(op))))"); $op(_u(a), b.v))
    @eval Base.$op(a::TypeTag, b::TypeTag) = (species_violation!("species-label arithmetic ($($(QuoteNode(op))))"); $op(a.v, b.v))
end

# ============================================================================
# SECTION 1c — DirTag: a direction tag for certifying point-group equivariance
# ============================================================================
# A lattice point-group element (90° rotation / reflection — the D4 of the square
# torus) acts on positions by MIXING coordinates, and a move written as
# `pos + (dr,dc)` with a HARDCODED offset is an ABSOLUTE constant that does not
# rotate, so such a move cannot be certified rotation-equivariant from one BFS
# (see doc/rotation-taint.md §1–§6). The fix (doc/rotation-taint.md §7): SUPPLY the
# direction set as a declared object `MOVES`, require moves to be expressed as
# `move(p, d)` with d drawn from `MOVES` via `rand_move!`, and tag the directions.
# A direction then becomes a covariant object the point group PERMUTES — exactly
# like a species label under a permutation. If
#   (i)  MOVES is closed under a point-group element g (a STATIC check), and
#   (ii) the BFS is CLEAN (every move went through move(p, ::DirTag) from
#        rand_move!; no bare position arithmetic; no absolute-position / species
#        misuse),
# then the algorithm is g-equivariant and the tau-BFS can be reduced over the
# (translation × species × point-group) orbit. Soundness rests on the same no-raw-
# unwrap discipline as tau (tau0) and species (t.v); the validation test in
# test_db.jl (rotation-reduced graph == direct build) backs it.
#
# A DirTag is OPAQUE: `move(p, ::DirTag)` is the only blessed use; any inspection
# (component access via indexing/iteration, comparison, arithmetic) is not
# overloaded and raises a MethodError, which the probe catches and treats as
# "decline rotation" — never a crash, never a silent pass.
struct DirTag; dr::Int; dc::Int; end

const _MOVESET = Ref(Tuple{Int,Int}[])                     # declared MOVES (set per run)

# rand_move!: pick a direction UNIFORMLY from the declared set (so the selection
# weight is point-group-invariant by construction). During the probe it returns a
# DirTag (opaque, covariant); otherwise a plain tuple (the fast/fallback path).
function rand_move!(rng)
    ms = _MOVESET[]
    isempty(ms) && cant("rand_move! called but no MOVES were declared")
    i = rand_choice_index!(rng, length(ms))
    _ROT_PROBE[] ? DirTag(ms[i][1], ms[i][2]) : ms[i]
end

# ---- particles / states ----
# Particle is parametric in the species-label type T so the SAME user code can run
# with plain Int labels (the fast default / concrete path) or with a tagged label
# (TypeTag, used only to certify species-equivariance during the tau-BFS — see
# SECTION 1b). PState is the supertype of both vector forms, so signatures and
# `return ... ::PState` accept either without change.
struct Particle{T}
    r::TauNum
    c::TauNum
    t::T
end
const PState = Vector{<:Particle}

# tau-augment a concrete (r,c,type) seed particle along both axes. Two flavours:
# the default uses a plain Int label (fast — used for the normal/fallback BFS); the
# tagged flavour wraps the label in a TypeTag so non-species-equivariant uses are
# detected (used only for the species-equivariance probe in SECTION 6).
aug_particle(r::Int, c::Int, t::Int)     = Particle(tau_r_aug(r), tau_c_aug(c), t)
aug_particle_tag(r::Int, c::Int, t::Int) = Particle(tau_r_aug(r), tau_c_aug(c), TypeTag(t))

# Extract the integer species label from either a plain Int or a TypeTag.
typeval(t::Int)::Int = t
typeval(t::TypeTag)::Int = t.v

# move(p, d): shift a particle by a direction d, the ONLY point-group-covariant way
# to move. Two flavours mirror the two label flavours:
#   * d::Tuple — the fast / fallback path (plain arithmetic). During the probe a
#     tuple direction means the algorithm bypassed rand_move! (used a hardcoded or
#     index-selected offset), which is NOT certifiable, so it is flagged.
#   * d::DirTag — the probe path: a declared, opaque direction. `_covshift` builds
#     the shifted coordinate via the TauNum CONSTRUCTOR (not `+`), so it does not
#     trip the bare-arithmetic rotation flag; the shift is covariant by contract.
_covshift(coord::TauNum, k::Int) = TauNum(coord.v + k, coord.cr, coord.cc, coord.tainted)
move(p::Particle, d::Tuple) =
    (_ROT_PROBE[] && rot_violation!("move with a non-declared direction (not from rand_move!)");
     Particle(p.r + d[1], p.c + d[2], p.t))
move(p::Particle, d::DirTag) = Particle(_covshift(p.r, d.dr), _covshift(p.c, d.dc), p.t)

# rev(d): the reverse of a direction (covariant — negation commutes with the point
# group). Lets an algorithm probe the backward displacement (e.g. VMMC's reverse
# energy) without breaking the point-group certificate, since -d is also covariant.
rev(d::Tuple)  = (-d[1], -d[2])
rev(d::DirTag) = DirTag(-d.dr, -d.dc)

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

# Is a next-state position a genuine translation-COVARIANT lattice position, i.e.
# does it shift by exactly the lattice offset (row tracks tau_r, col tracks tau_c,
# no nonlinear taint)?  Every output of a translation-invariant move must be of
# this form; anything else (an absolute coordinate from pmod, a reflection like
# -p.r, a nonlinear move like p.r^2) means the transition is NOT equivariant. We
# detect it here so the orbit-reduction optimisation is only ever applied to a
# genuinely equivariant algorithm — see SECTION 6.
is_covariant_pos(p::Particle) =
    p.r.cr == 1 && p.r.cc == 0 && !p.r.tainted &&
    p.c.cr == 0 && p.c.cc == 1 && !p.c.tainted

# ============================================================================
# SECTION 3 — BitSeqRNG and the symbolic random primitives
# ============================================================================
# Mirrors RunWithBitsAT in dbc_core.wl. Each BFS path replays the algorithm
# against a fixed bit string. Selection primitives consume bits and multiply
# the exact rational path coefficient; the Metropolis acceptance records a
# SYMBOLIC factor (the clamped Boltzmann threshold) rather than a float, so the
# downstream DB check is algebraically exact.

# Tau-violation is reported through a thread-local flag, reset per BFS run, so the
# geometry helpers above (which have no rng handle) can raise it. Thread-local so
# that a -parallel BFS over states does not race on it; in serial mode there is
# just one slot. `:static` scheduling keeps threadid() stable inside a BFS path
# (user code never yields), so threadid() indexing is safe here.
const _TAU     = [false]
const _TAU_MSG = [""]
tau_violation!(msg::String) = (i = Threads.threadid(); _TAU[i] = true; _TAU_MSG[i] = msg; nothing)

# --- OIP (order-independent iteration) controls --------------------------------
# `unordered(rng, items)` lets an algorithm declare an order-independent candidate
# loop: it yields a canonical order that reads NONE of the items' content (so it
# cannot itself break a species / point-group symmetry the way a sort tie-breaking
# on a label would) and consumes NO random bits (so it does not blow up the
# decision tree the way an explicit shuffle does). `_OIP_ORDER` selects which order
# to yield (1 = as given; 2 = reversed; 3 = cyclic shift), so the engine can re-BFS
# the representative in a second/third order and VERIFY the leaves are identical —
# the OIP cross-check that makes single-order execution sound (build_transitions).
# `_OIP_USED` (thread-local) records whether the algorithm called it this run.
const _OIP_ORDER = Ref(1)
const _OIP_USED  = [false]
_oip_clear!() = (for i in eachindex(_OIP_USED); _OIP_USED[i] = false; end; _OIP_ORDER[] = 1)
_oip_used()   = any(_OIP_USED)

# Resize the thread-local scratch (tau flags + interning caches) to the active
# thread count and clear it. Call once at the start of each run.
# Sized by maxthreadid() (not nthreads()): with `julia -t auto` the interactive
# thread pool means threadid() can exceed the default pool size.
function _init_threadlocal!()
    nt = Threads.maxthreadid()
    resize!(_TAU, nt);          fill!(_TAU, false)
    resize!(_TAU_MSG, nt);      fill!(_TAU_MSG, "")
    resize!(_SPECIES_BAD, nt);  fill!(_SPECIES_BAD, false)
    resize!(_SP_MSG, nt);       fill!(_SP_MSG, "")
    resize!(_ROT_BAD, nt);      fill!(_ROT_BAD, false)
    resize!(_ROT_MSG, nt);      fill!(_ROT_MSG, "")
    resize!(_OIP_USED, nt);     fill!(_OIP_USED, false); _OIP_ORDER[] = 1
    resize!(_TH_CACHES, nt)
    for i in 1:nt; _TH_CACHES[i] = Dict{Any,ThExpr}(); end
    nothing
end
_tau_any()  = any(_TAU)
_tau_first_msg() = (i = findfirst(!isempty, _TAU_MSG); i === nothing ? "" : _TAU_MSG[i])
_species_clear!() = (for i in eachindex(_SPECIES_BAD); _SPECIES_BAD[i] = false; _SP_MSG[i] = ""; end)
_species_any()  = any(_SPECIES_BAD)
_species_first_msg() = (i = findfirst(!isempty, _SP_MSG); i === nothing ? "" : _SP_MSG[i])
_rot_clear!() = (for i in eachindex(_ROT_BAD); _ROT_BAD[i] = false; _ROT_MSG[i] = ""; end)
_rot_any()  = any(_ROT_BAD)
_rot_first_msg() = (i = findfirst(!isempty, _ROT_MSG); i === nothing ? "" : _ROT_MSG[i])

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
const ExpVec = Vector{Q}                  # exponent L (linear in atoms) of exp(-beta L.J)
const Mono   = Vector{Int}                # monomial: integer power of each atom
const Poly   = Dict{Mono, Q}             # polynomial in the atoms (rational coeffs)
const BSum   = Dict{ExpVec, Poly}        # sum_L  poly_L(J) * exp(-beta L.J)

# ----------------------------------------------------------------------------
# Poly: the weight-coefficient ring.  Originally coefficients were rationals (Q);
# they are now multivariate polynomials in the coupling atoms so that weights may
# carry bare-coupling factors (e.g. a rate proportional to a field). A constant
# rational c is the degree-0 polynomial. The DB residual reduces to a sum
# sum_v p_v(J) exp(-beta v.J); distinct exp-monomials are linearly independent, so
# the residual vanishes on an open chamber iff every coefficient polynomial p_v is
# identically zero -- i.e. iff its Dict is empty after cancellation. That keeps the
# check exact: "all p_v == 0" is just "every monomial coefficient is 0 in Q".
# ----------------------------------------------------------------------------
poly_iszero(p::Poly) = isempty(p)
poly_const(c::Q, nA::Int) = c == 0 ? Poly() : Poly(zeros(Int, nA) => c)
function poly_add!(p::Poly, m::Mono, c::Q)
    c == 0 && return p
    v = get(p, m, Q(0)) + c
    v == 0 ? delete!(p, m) : (p[m] = v); p
end
poly_add(a::Poly, b::Poly) = (r = copy(a); for (m,c) in b; poly_add!(r,m,c); end; r)
poly_neg(p::Poly) = Poly(m => -c for (m,c) in p)
function poly_mul(a::Poly, b::Poly)
    (isempty(a) || isempty(b)) && return Poly()
    r = Poly(); for (ma,ca) in a, (mb,cb) in b; poly_add!(r, ma .+ mb, ca*cb); end; r
end
# (is p a degree-0 constant?, its value) -- a polynomial is constant iff its only
# monomial is the all-zero one.
poly_isconst(p::Poly) = isempty(p) ? (true, Q(0)) :
    (length(p) == 1 && all(==(0), first(keys(p)))) ? (true, first(values(p))) : (false, Q(0))
# the degree-1 polynomial  sum_a L[a] * J_a
function poly_linear(L::ExpVec, nA::Int)
    p = Poly()
    for a in 1:nA
        L[a] == 0 && continue
        m = zeros(Int, nA); m[a] = 1; p[m] = L[a]
    end
    p
end

# ---- BSum: Laurent polynomial in exp-monomials, Poly coefficients ----
function bs_add!(s::BSum, L::ExpVec, p::Poly)
    poly_iszero(p) && return s
    q = haskey(s, L) ? poly_add(s[L], p) : copy(p)
    poly_iszero(q) ? delete!(s, L) : (s[L] = q); s
end
bs_const(c::Q, nA::Int) = c == 0 ? BSum() : BSum(zeros(Q, nA) => poly_const(c, nA))
bs_poly(p::Poly, nA::Int) = poly_iszero(p) ? BSum() : BSum(zeros(Q, nA) => p)  # pure polynomial
bs_addsum(a::BSum, b::BSum) = (r = copy(a); for (L,p) in b; bs_add!(r,L,p); end; r)
bs_sub(a::BSum, b::BSum)    = (r = copy(a); for (L,p) in b; bs_add!(r,L,poly_neg(p)); end; r)
function bs_mul(a::BSum, b::BSum)
    r = BSum()
    for (La,pa) in a, (Lb,pb) in b; bs_add!(r, La .+ Lb, poly_mul(pa,pb)); end
    r
end
bs_shift(a::BSum, E::ExpVec) = BSum((L .+ E) => copy(p) for (L,p) in a)

# Val = num / prod(den), where each den factor is itself a BSum (a binomial such as
# 1 - exp(-beta L) or 1 + exp(-beta L), or any 1- or 2-term exp-polynomial with
# CONSTANT coefficients). Storing the factor explicitly (not just L) is what lets
# the engine divide by 1+exp (Barker/Glauber) and other sign-definite binomials.
struct Val
    num::BSum
    den::Vector{BSum}
end
val_const(c::Q, nA::Int)  = Val(bs_const(c, nA), BSum[])
val_boltz(L::ExpVec, nA::Int) = Val(BSum(copy(L) => poly_const(Q(1), nA)), BSum[])  # exp(-beta L)
val_linear(L::ExpVec, nA::Int) = Val(bs_poly(poly_linear(L, nA), nA), BSum[])       # <L, J>
val_mul(a::Val, b::Val)   = Val(bs_mul(a.num, b.num), vcat(a.den, b.den))

# The binomial factor 1 - exp(-beta L) (the classic VMMC denominator).
binom_oneminus(L::ExpVec, nA::Int) =
    BSum(zeros(Q, nA) => poly_const(Q(1), nA), copy(L) => poly_const(Q(-1), nA))

expand_binoms(dens::Vector{BSum}, nA::Int) =
    (acc = bs_const(Q(1), nA); for f in dens; acc = bs_mul(acc, f); end; acc)
val_oneminus(v::Val, nA::Int) = Val(bs_sub(expand_binoms(v.den, nA), v.num), copy(v.den))

# multiset union (max multiplicity) / difference over den factors (BSums, by value)
function ms_unionmax(a::Vector{BSum}, b::Vector{BSum})
    cnt = Dict{BSum,Int}()
    for f in a; cnt[f] = max(get(cnt,f,0), count(==(f), a)); end
    for f in b; cnt[f] = max(get(cnt,f,0), count(==(f), b)); end
    out = BSum[]; for (f,k) in cnt, _ in 1:k; push!(out, f); end; out
end
function ms_diff(big::Vector{BSum}, small::Vector{BSum})
    rem = copy(big)
    for f in small
        i = findfirst(==(f), rem); i === nothing && cant("denominator multiset diff failed"); deleteat!(rem, i)
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
struct ThConst  <: ThExpr; c::Q; end
struct ThBoltz  <: ThExpr; L::RatForm; end                 # exp(-beta * L)
struct ThLinear <: ThExpr; L::RatForm; end                 # the bare value <L, J> (polynomial weight)
struct ThOp     <: ThExpr; op::Symbol; a::ThExpr; b::ThExpr; end   # :+,:-,:*,:/
struct ThMin    <: ThExpr; a::ThExpr; b::ThExpr; end
struct ThMax    <: ThExpr; a::ThExpr; b::ThExpr; end
struct ThPiece  <: ThExpr; clauses::Vector{Tuple{Vector{Cond},ThExpr}}; default::ThExpr; end

# Hash-consing (interning): structurally-equal thresholds share ONE object, so
# the millions of thresholds built during the BFS collapse to a few thousand
# canonical nodes. This makes weight-deduplication an objectid comparison
# instead of a deep structural hash (the dominant cost otherwise). Keys are
# built from already-interned children by objectid, so they stay small.
const _TH_CACHES = [Dict{Any,ThExpr}()]     # one interning table per thread
_intern(make::Function, key) = get!(make, _TH_CACHES[Threads.threadid()], key)

# Translation-facing builders (energies are LinForm so tau is tracked).
th_const(x)                 = _intern((:c, Q(x))) do; ThConst(Q(x)) end
th_boltz(L::LinForm)        = (rl = tau0_checked(L); _intern((:b, _form_key(rl))) do; ThBoltz(rl) end)
th_linear(L::LinForm)       = (rl = tau0_checked(L); _intern((:lin, _form_key(rl))) do; ThLinear(rl) end)
th_add(a::ThExpr,b::ThExpr) = _intern((:op,:+,objectid(a),objectid(b))) do; ThOp(:+,a,b) end
th_sub(a::ThExpr,b::ThExpr) = _intern((:op,:-,objectid(a),objectid(b))) do; ThOp(:-,a,b) end
th_mul(a::ThExpr,b::ThExpr) = _intern((:op,:*,objectid(a),objectid(b))) do; ThOp(:*,a,b) end
th_div(a::ThExpr,b::ThExpr) = _intern((:op,:/,objectid(a),objectid(b))) do; ThOp(:/,a,b) end
th_min(a::ThExpr,b::ThExpr) = _intern((:min,objectid(a),objectid(b))) do; ThMin(a,b) end
th_max(a::ThExpr,b::ThExpr) = _intern((:max,objectid(a),objectid(b))) do; ThMax(a,b) end
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

# Order-independent iteration (the OIP contract). Yields `items` in a canonical
# order that reads NONE of their content, so it cannot itself break a species /
# point-group symmetry (unlike a sort tie-breaking on a label), and consumes NO
# random bits, so it does NOT blow up the decision tree the way an explicit shuffle
# (Fisher-Yates over the rng) does. The author asserts the loop body's effect is
# INDEPENDENT of the visitation order; build_transitions VERIFIES this per run by
# re-BFSing the representative in a second/third order and asserting an identical
# leaf multiset (the OIP cross-check) — a mismatch is a hard error. `rng` is taken
# for call-site symmetry with the other primitives but is not consumed. Returns a
# fresh Vector (the caller may iterate or mutate it).
function unordered(rng, items::AbstractVector)
    i = Threads.threadid(); (i <= length(_OIP_USED)) && (_OIP_USED[i] = true)
    v = collect(items)
    ord = _OIP_ORDER[]
    ord == 1 && return v
    ord == 2 && return reverse(v)
    length(v) <= 1 ? v : vcat(v[2:end], v[1:1])        # ord == 3: cyclic shift by 1
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
    cs = NTuple{3,Int}[(mod(intval(p.r) - 1, n) + 1, mod(intval(p.c) - 1, n) + 1, typeval(p.t))
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
    tagged = eltype(seed) <: Particle{TypeTag}      # species probe? (then check covariance)
    while !isempty(queue)
        bits = popfirst!(queue)
        rng  = BitSeqRNG(bits)
        try
            nxt = algo(rng, seed)::PState
            for p in nxt
                is_covariant_pos(p) ||
                    tau_violation!("a next-state position is not a pure lattice translation " *
                                   "of the input (absolute, reflected, or nonlinear move)")
                # species-covariance: an output label must be an INHERITED tag, not a
                # fresh literal (which would not relabel under a species permutation).
                tagged && !(p.t isa TypeTag) &&
                    species_violation!("a next-state species label is a fresh literal, " *
                                       "not an inherited label (non-equivariant)")
            end
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
    elseif t isa ThLinear; _rfval(t.L, J)
    elseif t isa ThOp
        a = eval_th_float(t.a,J,beta); b = eval_th_float(t.b,J,beta)
        t.op === :+ ? a+b : t.op === :- ? a-b : t.op === :* ? a*b : a/b
    elseif t isa ThMin; min(eval_th_float(t.a,J,beta), eval_th_float(t.b,J,beta))
    elseif t isa ThMax; max(eval_th_float(t.a,J,beta), eval_th_float(t.b,J,beta))
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

# Concrete state -> tau-FREE PState with Int labels (for energy of real states).
concrete_pstate(cs::CState)::PState = Particle{Int}[Particle(TauNum(r), TauNum(c), t) for (r, c, t) in cs]
# Concrete state -> tau-AUGMENTED PState with Int labels (the normal BFS seed).
augmented_pstate(cs::CState)::PState = Particle{Int}[aug_particle(r, c, t) for (r, c, t) in cs]
# Same, but with TAGGED labels (used only for the species-equivariance probe).
augmented_pstate_tag(cs::CState)::PState = Particle{TypeTag}[aug_particle_tag(r, c, t) for (r, c, t) in cs]

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

# Square-lattice point-group actions on a concrete state. Together these cover all
# 8 elements of D4 (the full point group of the square lattice) as individual
# candidates, not just the two generators. Every lattice isometry is here so any
# subgroup -- D4, D2, C4, C2, or a single reflection -- can be discovered by the
# graph-verification step without assuming anything about the algorithm. All are
# lattice isometries and preserve squared minimum-image distances. They are used
# ONLY to reduce the detailed-balance pair check, and ONLY after being VERIFIED to
# be symmetries of the already-computed transition graph and energy (SECTION 7).
#
# Coordinate convention: (r,c) with r = row (1..n), c = column (1..n).
# rotate90:   (r,c) → (c, n+1-r)   [90° CCW rotation]
# rotate180:  (r,c) → (n+1-r, n+1-c)
# rotate270:  (r,c) → (n+1-c, r)   [90° CW = 270° CCW]
# reflect:    (r,c) → (c, r)        [main diagonal r=c]
# reflect_h:  (r,c) → (n+1-r, c)   [horizontal axis, i.e. flip rows]
# reflect_v:  (r,c) → (r, n+1-c)   [vertical axis,   i.e. flip cols]
# reflect_ad: (r,c) → (n+1-c, n+1-r) [anti-diagonal]
rotate_cstate(cs::CState, n::Int)::CState =
    sort(NTuple{3,Int}[(c, n + 1 - r, t) for (r, c, t) in cs])
rotate180_cstate(cs::CState, n::Int)::CState =
    sort(NTuple{3,Int}[(n + 1 - r, n + 1 - c, t) for (r, c, t) in cs])
rotate270_cstate(cs::CState, n::Int)::CState =
    sort(NTuple{3,Int}[(n + 1 - c, r, t) for (r, c, t) in cs])
reflect_cstate(cs::CState, n::Int)::CState =
    sort(NTuple{3,Int}[(c, r, t) for (r, c, t) in cs])
reflect_h_cstate(cs::CState, n::Int)::CState =
    sort(NTuple{3,Int}[(n + 1 - r, c, t) for (r, c, t) in cs])
reflect_v_cstate(cs::CState, n::Int)::CState =
    sort(NTuple{3,Int}[(r, n + 1 - c, t) for (r, c, t) in cs])
reflect_ad_cstate(cs::CState, n::Int)::CState =
    sort(NTuple{3,Int}[(n + 1 - c, n + 1 - r, t) for (r, c, t) in cs])

# The point group as (name, cstate-action, direction-linear-part). The cstate
# action transforms a whole concrete state; the linear part transforms a (dr,dc)
# DIRECTION (the homogeneous, translation-free part of the same isometry). Identity
# is first. These pair up exactly: applying the cstate action to a state moved by d
# equals applying it to the state and moving by lin(d). That correspondence is why
# "MOVES closed under lin" + "clean probe" ⟹ the cstate action is a graph symmetry.
const _POINTGROUP = [
    ("identity",   (cs,n)->cs,                  ((dr,dc),)->(dr, dc)),
    ("rotate90",   rotate_cstate,               ((dr,dc),)->(dc, -dr)),
    ("rotate180",  rotate180_cstate,            ((dr,dc),)->(-dr, -dc)),
    ("rotate270",  rotate270_cstate,            ((dr,dc),)->(-dc, dr)),
    ("reflect",    reflect_cstate,              ((dr,dc),)->(dc, dr)),
    ("reflect_h",  reflect_h_cstate,            ((dr,dc),)->(-dr, dc)),
    ("reflect_v",  reflect_v_cstate,            ((dr,dc),)->(dr, -dc)),
    ("reflect_ad", reflect_ad_cstate,           ((dr,dc),)->(-dc, -dr)),
]

# The SUBGROUP of the point group under which the declared direction set is closed.
# Returns the indices into _POINTGROUP (always includes 1 = identity). This is the
# STATIC half of the rotation certificate: a property of the params, provable
# without running R·s. The closed set is automatically a subgroup (closed under
# composition and inverse), and each member is a genuine symmetry once the probe
# is clean. An empty MOVES (no position moves, e.g. a pure type-swap) is vacuously
# closed under all of D4.
function pointgroup_subgroup(moves::Vector{Tuple{Int,Int}})::Vector{Int}
    S = Set(moves)
    idxs = Int[]
    for (i, (_, _, lin)) in enumerate(_POINTGROUP)
        all(lin(d) in S for d in moves) && push!(idxs, i)
    end
    idxs
end
pt_apply(cs::CState, gi::Int, n::Int)::CState = _POINTGROUP[gi][2](cs, n)
pt_name(gi::Int) = _POINTGROUP[gi][1]

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

# Relabel a concrete state's species by sigma.
relabel_cstate(cs::CState, σ::Dict{Int,Int})::CState =
    sort(NTuple{3,Int}[(r, c, σ[t]) for (r, c, t) in cs])

# All species permutations preserving the type-multiset (identity first). For
# all-distinct labels this is the full symmetric group; for repeated labels it is
# the product of symmetric groups over equal-multiplicity classes.
function _all_perms(v::Vector{Int})
    length(v) <= 1 && return [v]
    out = Vector{Int}[]
    for i in eachindex(v)
        for p in _all_perms(v[[j for j in eachindex(v) if j != i]]); push!(out, vcat(v[i], p)); end
    end
    out
end
function _type_group_full(typemult::Vector{Int})
    cnt = Dict{Int,Int}(); for t in typemult; cnt[t] = get(cnt,t,0)+1; end
    labels = sort(collect(keys(cnt)))
    grp = Dict{Int,Int}[]
    for p in _all_perms(labels)
        σ = Dict(labels[i] => p[i] for i in eachindex(labels))
        all(cnt[t] == cnt[σ[t]] for t in labels) && push!(grp, σ)
    end
    id = Dict(t => t for t in labels)
    sort!(grp; by = σ -> (σ == id ? 0 : 1))      # identity first
    grp
end

# Orbits of the translation representatives under the species group. Returns, for
# each translation rep `tr`, a pair (combined_rep, k) with combined_rep a chosen
# translation rep and k the index in σgroup such that
# repof_t[relabel(combined_rep, σ_k)] == tr; plus the list of combined reps.
function combined_trep_orbits(treps::Vector{CState}, repof_t::Dict{CState,CState},
                              σgroup::Vector{Dict{Int,Int}})
    crep_of = Dict{CState,CState}(); k_of = Dict{CState,Int}(); creps = CState[]
    for tr in treps
        haskey(crep_of, tr) && continue
        push!(creps, tr)
        for (k, σ) in enumerate(σgroup)
            tr2 = repof_t[relabel_cstate(tr, σ)]
            if !haskey(crep_of, tr2); crep_of[tr2] = tr; k_of[tr2] = k; end
        end
    end
    creps, crep_of, k_of
end

# Orbits of the translation reps under the FULL reduction group: species
# permutations σ TIMES the point-group subgroup H (indices Hidx into _POINTGROUP).
# Returns the combined reps and, per trep `tr`, a pair (combined_rep, (k, gi)) with
#     repof_t[ pt_apply(relabel(combined_rep, σ_k), gi) ] == tr.
# Species relabel and point-group action commute (labels vs positions), so the
# composite is well defined. With σgroup=[identity] and Hidx=[1] this degenerates
# to the plain translation reduction; with Hidx=[1] to combined_trep_orbits.
function combined_trep_orbits_pg(treps::Vector{CState}, repof_t::Dict{CState,CState},
                                 σgroup::Vector{Dict{Int,Int}}, Hidx::Vector{Int}, n::Int)
    crep_of = Dict{CState,CState}(); kg_of = Dict{CState,Tuple{Int,Int}}(); creps = CState[]
    for tr in treps
        haskey(crep_of, tr) && continue
        push!(creps, tr)
        for (k, σ) in enumerate(σgroup), gi in Hidx
            tr2 = repof_t[ pt_apply(relabel_cstate(tr, σ), gi, n) ]
            if !haskey(crep_of, tr2); crep_of[tr2] = tr; kg_of[tr2] = (k, gi); end
        end
    end
    creps, crep_of, kg_of
end

struct BFSResult
    states       :: Vector{CState}
    idx          :: Dict{CState,Int}
    uweights     :: Vector{Leaf}                 # unique (representative) leaf weights
    trans        :: Vector{Tuple{Int,Int,Int}}   # (src, dst, weight-index), src != dst
    tau_free     :: Bool
    tau_msg      :: String
    n            :: Int                          # lattice side (for symmetry actions)
    species_free :: Bool                         # species-equivariant BFS reduction used?
    nbfs         :: Int                          # number of states actually BFS'd
    pg_idx       :: Vector{Int}                  # point-group elements used (indices into
                                                 # _POINTGROUP); empty if none beyond identity
end

# BFS a list of seed states, optionally across threads. Each thread uses its own
# interning cache and tau flag (set up by _init_threadlocal!), so the only shared
# output is the per-seed leaf vector written to a preallocated slot. A CantHandle
# raised inside a worker is unwrapped and rethrown so check.jl can report it.
function _bfs_seeds(algo, seeds::Vector{CState}, n::Int, maxdepth::Int, parallel::Bool;
                   seedfn = augmented_pstate)
    out = Vector{Vector{Leaf}}(undef, length(seeds))
    if parallel && Threads.nthreads() > 1
        try
            Threads.@threads :static for i in 1:length(seeds)
                out[i] = build_state_leaves(algo, seedfn(seeds[i]), n, maxdepth)
            end
        catch e
            throw(_unwrap_cant(e))
        end
    else
        for i in 1:length(seeds)
            out[i] = build_state_leaves(algo, seedfn(seeds[i]), n, maxdepth)
        end
    end
    out
end
# Dig a CantHandle out of a (possibly nested) TaskFailedException.
function _unwrap_cant(e)
    e isa CantHandle && return e
    if e isa TaskFailedException; return _unwrap_cant(e.task.exception); end
    if e isa CompositeException && !isempty(e.exceptions); return _unwrap_cant(e.exceptions[1]); end
    e
end

# ---- OIP cross-check: verify order-independence EXACTLY (no floats) -------------
# A canonical, deterministic key for a threshold (reusing the canonical RatForm /
# condition keys), so structurally-equal thresholds get equal keys regardless of how
# they were built — the comparison is independent of the interning cache state.
function _th_key(t::ThExpr)
    t isa ThConst  ? (:const, t.c) :
    t isa ThBoltz  ? (:boltz, _form_key(t.L)) :
    t isa ThLinear ? (:lin,   _form_key(t.L)) :
    t isa ThOp     ? (:op, t.op, _th_key(t.a), _th_key(t.b)) :
    t isa ThMin    ? (:min, _th_key(t.a), _th_key(t.b)) :
    t isa ThMax    ? (:max, _th_key(t.a), _th_key(t.b)) :
    t isa ThPiece  ? (:piece, Tuple((Tuple(cond_key(c) for c in gs), _th_key(v)) for (gs, v) in t.clauses), _th_key(t.default)) :
    error("unknown ThExpr in _th_key")
end
# A leaf's order-invariant signature: successor, rational coefficient, and the
# MULTISET of acceptance factors (a leaf weight is a PRODUCT, so factor order is
# irrelevant). Factors are canonicalised to strings so a heterogeneous multiset
# sorts without ambiguity.
_leafsig(lf::Leaf) = (lf.next, lf.coeff,
    sort(String[string(_th_key(f.thr)) * (f.accepted ? "+" : "-") for f in lf.factors]))
# Do two leaf sets of the SAME seed agree on their OFF-DIAGONAL transitions?
#
# The diagonal (self-loop, next == seed) carries an algorithm's rejection
# probability, which an early-abort move (e.g. VMMC's frustration) decomposes into
# DIFFERENT partial-product leaves per visiting order. But the diagonal enters
# NEITHER detailed balance NOR global balance (both cancel the s==t term; the stored
# transition graph already drops self-loops), and the OFF-diagonal transition leaves
# are order-invariant EXACTLY (any order links the same candidates with the same
# commutative product). So comparing the off-diagonal leaf multiset is an exact,
# float-free order-independence test that is sufficient for either verdict — and
# tight: an order-DEPENDENT successor probability changes the off-diagonal multiset.
function _oip_match(la::Vector{Leaf}, lb::Vector{Leaf}, seed::CState)::Bool
    da = Dict{Any,Int}()
    for lf in la; lf.next == seed && continue; k = _leafsig(lf); da[k] = get(da, k, 0) + 1; end
    db = Dict{Any,Int}()
    for lf in lb; lf.next == seed && continue; k = _leafsig(lf); db[k] = get(db, k, 0) + 1; end
    da == db
end

# BFS `seeds` in canonical order; if the algorithm used `unordered`, ALSO BFS in a
# reversed and a cyclically-shifted order and assert each seed's leaf multiset is
# identical (the exact OIP cross-check). A mismatch means the loop body is NOT
# order-independent, so `unordered` was misused — a hard error (the single-order
# leaves would be a different algorithm than intended). The alt passes use plain Int
# seeds with the rotation probe OFF and restore the species/rotation flags, so they
# are purely a leaf comparison and never perturb the symmetry certificates. Returns
# the canonical-order leaves.
function _bfs_seeds_oip(algo, seeds::Vector{CState}, n::Int, maxdepth::Int, parallel::Bool;
                       seedfn = augmented_pstate)
    _OIP_ORDER[] = 1
    base = _bfs_seeds(algo, seeds, n, maxdepth, parallel; seedfn = seedfn)
    if _oip_used()
        sp = copy(_SPECIES_BAD); spm = copy(_SP_MSG)
        ro = copy(_ROT_BAD);     rom = copy(_ROT_MSG); rp = _ROT_PROBE[]
        _ROT_PROBE[] = false
        # One reversed order: with the candidate list reversed, two genuinely
        # order-dependent prefixes differ, while an order-independent body's summed
        # per-successor probabilities are unchanged. (A reversal already exercises
        # the cyclic-shift order on >2 candidates; order 3 is available for a
        # stronger check if ever needed.)
        _OIP_ORDER[] = 2
        alt = _bfs_seeds(algo, seeds, n, maxdepth, parallel; seedfn = augmented_pstate)
        for i in eachindex(seeds)
            _oip_match(base[i], alt[i], seeds[i]) || begin
                _OIP_ORDER[] = 1; _ROT_PROBE[] = rp
                cant("unordered(): the result depends on the candidate visitation order " *
                     "(off-diagonal transition probabilities differ between two orders) — " *
                     "`unordered` requires the loop body's effect to be independent of order")
            end
        end
        _OIP_ORDER[] = 1; _ROT_PROBE[] = rp
        copyto!(_SPECIES_BAD, sp); copyto!(_SP_MSG, spm)
        copyto!(_ROT_BAD, ro);     copyto!(_ROT_MSG, rom)
    end
    base
end

function build_transitions(algo, energy, states::Vector{CState}, n::Int, maxdepth::Int;
                           parallel::Bool=false, species::Bool=true,
                           moves::Union{Nothing,Vector{Tuple{Int,Int}}}=nothing,
                           use_pointgroup::Bool=true)::BFSResult
    _init_threadlocal!()                    # fresh per-thread interning + tau/species/rot flags
    idx = Dict(cs => i for (i, cs) in enumerate(states))
    uweights = Leaf[]; uw_idx = Dict{Any,Int}()
    widx!(lf::Leaf) = get!(uw_idx, _weight_key(lf)) do; push!(uweights, lf); length(uweights) end
    trans = Tuple{Int,Int,Int}[]

    treps, repof, gof = translation_orbits(states, n)
    fullσ  = _type_group_full(Int[t for (r,c,t) in states[1]])      # identity first
    σgroup = species ? fullσ : Dict{Int,Int}[fullσ[1]]
    rep_leaves = Dict{CState,Vector{Leaf}}()

    # Point-group: the static subgroup H under which the declared MOVES are closed.
    # `_MOVESET` is still set when MOVES are declared (so rand_move!/move work on the
    # normal path) even with `use_pointgroup=false`, which forces H = {identity} —
    # the validation baseline (same algorithm, no point-group reduction).
    if moves === nothing
        _MOVESET[] = Tuple{Int,Int}[]; Hidx = Int[1]
    else
        _MOVESET[] = moves; Hidx = use_pointgroup ? pointgroup_subgroup(moves) : Int[1]
    end

    # --- general derivation: build every state's transitions from the combined reps
    # by translate ∘ point-group ∘ species-relabel. Point-group preserves distances
    # and types so it leaves WEIGHTS unchanged; only the species relabel permutes
    # atoms (permwi). cr_of: trep->combined-rep, kg_of: trep->(σ-index, pg-index).
    function derive!(creps, cr_of, kg_of, σg)
        wi_of = IdDict{Leaf,Int}()
        for cr in creps, lf in rep_leaves[cr]; wi_of[lf] = widx!(lf); end
        wcache = Dict{Tuple{Int,Int},Int}()                 # (base wi, σ-index k) -> wi
        permwi(wi::Int, lf::Leaf, k::Int) = k == 1 ? wi : get!(wcache, (wi, k)) do
            σ = σg[k]
            widx!(Leaf(lf.next, lf.coeff,
                       ThFactor[ThFactor(permute_atoms_th(f.thr, σ), f.accepted) for f in lf.factors]))
        end
        for (s, si) in idx
            tr = repof[s]; cr = cr_of[tr]; (k, gi) = kg_of[tr]; σ = σg[k]
            base = pt_apply(relabel_cstate(cr, σ), gi, n)   # == translate(tr, a)
            a = gof[base]
            dr = mod(gof[s][1] - a[1], n); dc = mod(gof[s][2] - a[2], n)
            for lf in rep_leaves[cr]
                nxt = translate_cstate(pt_apply(relabel_cstate(lf.next, σ), gi, n), dr, dc, n)
                dst = idx[nxt]
                dst != si && push!(trans, (si, dst, permwi(wi_of[lf], lf, k)))
            end
        end
    end

    # --- attempt the COMBINED (translation × species × point-group) BFS reduction.
    # BFS one rep per FULL combined orbit using tagged labels (certify species) and
    # tagged directions / rotation-arithmetic flagging (certify point-group), all
    # from a single BFS. We then DOWNGRADE to whatever was actually certified
    # (species and/or point-group), BFS-ing only the extra reps the coarser orbit
    # needs (already-BFS'd reps are reused), and derive everything from them.
    want_probe = length(σgroup) > 1 || length(Hidx) > 1
    if want_probe
        creps0, _, _ = combined_trep_orbits_pg(treps, repof, σgroup, Hidx, n)
        _species_clear!(); _rot_clear!()
        seedfn = length(σgroup) > 1 ? augmented_pstate_tag : augmented_pstate
        _ROT_PROBE[] = length(Hidx) > 1
        probe_ok = true
        try
            vec = _bfs_seeds_oip(algo, creps0, n, maxdepth, parallel; seedfn = seedfn)
            for i in eachindex(creps0); rep_leaves[creps0[i]] = vec[i]; end
        catch e
            (e isa CantHandle) && rethrow(e)
            probe_ok = false; empty!(rep_leaves); _init_threadlocal!()   # discard probe state
        finally
            _ROT_PROBE[] = false
        end
        if probe_ok && !_tau_any()
            use_species = length(σgroup) > 1 && !_species_any()
            use_pg      = length(Hidx)  > 1 && !_rot_any()
            σg = use_species ? σgroup : Dict{Int,Int}[σgroup[1]]
            Hi = use_pg ? Hidx : Int[1]
            creps, cr_of, kg_of = combined_trep_orbits_pg(treps, repof, σg, Hi, n)
            need = CState[cr for cr in creps if !haskey(rep_leaves, cr)]
            if !isempty(need)
                extra = _bfs_seeds_oip(algo, need, n, maxdepth, parallel)   # Int seeds, no probe
                for i in eachindex(need); rep_leaves[need[i]] = extra[i]; end
            end
            derive!(creps, cr_of, kg_of, σg)
            return BFSResult(states, idx, uweights, trans, true, "", n,
                             use_species, length(creps), use_pg ? Hidx : Int[])
        end
        # Not certified at all (tau flagged): keep any BFS'd leaves and fall through.
    end

    # --- translation-only reduction / all-states fallback -----------------------
    need = CState[tr for tr in treps if !haskey(rep_leaves, tr)]
    extra = _bfs_seeds_oip(algo, need, n, maxdepth, parallel)
    for i in eachindex(need); rep_leaves[need[i]] = extra[i]; end
    tau_free = !_tau_any(); tau_msg = _tau_first_msg()

    if tau_free
        wi_of = IdDict{Leaf,Int}()
        for tr in treps, lf in rep_leaves[tr]; wi_of[lf] = widx!(lf); end
        for (s, si) in idx
            tr = repof[s]; (dr, dc) = gof[s]
            for lf in rep_leaves[tr]
                dst = idx[translate_cstate(lf.next, dr, dc, n)]
                dst != si && push!(trans, (si, dst, wi_of[lf]))
            end
        end
        return BFSResult(states, idx, uweights, trans, true, tau_msg, n, false, length(treps), Int[])
    end

    # Not translation invariant: direct BFS from EVERY state (no equivariance).
    moreneed = CState[s for s in states if !haskey(rep_leaves, s)]
    moreextra = _bfs_seeds_oip(algo, moreneed, n, maxdepth, parallel)
    for i in eachindex(moreneed); rep_leaves[moreneed[i]] = moreextra[i]; end
    for (s, si) in idx
        for lf in rep_leaves[s]
            wi = widx!(lf); dst = idx[lf.next]
            dst != si && push!(trans, (si, dst, wi))
        end
    end
    BFSResult(states, idx, uweights, trans, false, tau_msg, n, false, length(states), Int[])
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

# Unique-weight key: thresholds are interned (hash-consed), so structurally
# equal thresholds are the SAME object and objectid identifies them in O(1).
_weight_key(lf::Leaf) = (lf.coeff, Tuple((objectid(f.thr), f.accepted) for f in lf.factors))

vecof(f::RatForm, aidx::Dict{Atom,Int}, nA::Int) =
    (v = zeros(Q, nA); for (a, c) in f; v[aidx[a]] = c; end; v)

# ---- atom collection: every coupling that appears anywhere ----
function _collect_atoms_th!(set::Set{Atom}, t::ThExpr)
    if t isa ThConst
    elseif t isa ThBoltz; for a in keys(t.L); push!(set,a); end
    elseif t isa ThLinear; for a in keys(t.L); push!(set,a); end
    elseif t isa ThOp; _collect_atoms_th!(set,t.a); _collect_atoms_th!(set,t.b)
    elseif t isa ThMin || t isa ThMax; _collect_atoms_th!(set,t.a); _collect_atoms_th!(set,t.b)
    else; for (gs,v) in t.clauses; for g in gs, a in keys(g.lhs); push!(set,a); end; _collect_atoms_th!(set,v); end; _collect_atoms_th!(set,t.default)
    end
end

# ---- "static" Val evaluation of a condition-free ThExpr (ThMin operands) ----
# Errors if it meets a ThPiece/ThMin, i.e. the operands of a Min must not
# themselves contain conditions (true for VMMC; fail-loud otherwise).
function eval_static(t::ThExpr, aidx, nA)::Val
    if t isa ThConst; val_const(t.c, nA)
    elseif t isa ThBoltz; val_boltz(to_vec(t.L, aidx, nA), nA)
    elseif t isa ThLinear; val_linear(to_vec(t.L, aidx, nA), nA)
    elseif t isa ThOp
        a = eval_static(t.a,aidx,nA); b = eval_static(t.b,aidx,nA)
        t.op === :+ ? val_add(a,b,nA) : t.op === :- ? val_sub(a,b,nA) :
        t.op === :* ? val_mul(a,b)    : val_div(a,b)
    else
        cant("Min/Max over a condition-bearing expression is not supported (nested min/max)")
    end
end

# Condition implied by Min[a,b] (TRUE iff a < b), as (eff_lhs, strict).
# Requires the cleared-denominator difference a-b to be a balanced binomial
# c*(exp(-bL1) - exp(-bL2)); then a<b reduces to the linear form (L1-L2)>0.
#
# SOUNDNESS ASSUMPTION: clearing the operands' (1-exp) denominators preserves
# the inequality direction only where those denominators are POSITIVE. This
# holds wherever the Min is actually evaluated, because (as in VMMC) the Min
# sits under a guard that forces denominator positivity (e.g. eInit<eFwd makes
# 1-exp(-b*(eFwd-eInit)) > 0). The registered hyperplane is correct in that
# region; in chambers where the guard fails the Min is never reached, so the
# condition's value there is irrelevant. A Min used WITHOUT such a guard would
# need explicit denominator-sign tracking (not implemented) -- see AUDIT.md.
function min_condition(a::ThExpr, b::ThExpr, aidx, nA)
    va = eval_static(a, aidx, nA); vb = eval_static(b, aidx, nA)
    numer = bs_sub(bs_mul(va.num, expand_binoms(vb.den, nA)),
                   bs_mul(vb.num, expand_binoms(va.den, nA)))   # numerator of (a-b)
    isempty(numer) && return nothing                            # a == b, no condition
    length(numer) == 2 || cant("Min/Max condition is not a linear hyperplane (numerator has $(length(numer)) exp-terms)")
    (k1,p1),(k2,p2) = collect(numer)
    o1,c1 = poly_isconst(p1); o2,c2 = poly_isconst(p2)
    (o1 && o2) || cant("Min/Max condition has a non-constant (polynomial) coefficient — switch is not a hyperplane")
    c1 == -c2 || cant("Min/Max condition numerator is an unbalanced binomial (not a hyperplane)")
    kp, km = c1 > 0 ? (k1,k2) : (k2,k1)                         # +coeff key, -coeff key
    # a<b  <=>  numer<0  <=>  exp(-b*kp) < exp(-b*km)  <=>  kp>km  <=>  (km-kp)<0
    eff = kp .- km                                              # sigma=1 (a<b) iff eff.J > 0
    (eff, true)
end

# ---- exact Val evaluation of a ThExpr under a chamber sign pattern ----
function eval_val(t::ThExpr, σ::Vector{Int}, ctx, aidx, nA)::Val
    if t isa ThConst; val_const(t.c, nA)
    elseif t isa ThBoltz; val_boltz(to_vec(t.L, aidx, nA), nA)
    elseif t isa ThLinear; val_linear(to_vec(t.L, aidx, nA), nA)
    elseif t isa ThOp
        a = eval_val(t.a,σ,ctx,aidx,nA); b = eval_val(t.b,σ,ctx,aidx,nA)
        t.op === :+ ? val_add(a,b,nA) : t.op === :- ? val_sub(a,b,nA) :
        t.op === :* ? val_mul(a,b)    : val_div(a,b)
    elseif t isa ThMin
        idx = ctx.thmin[objectid(t)]                            # sigma=1 iff a<b
        idx == 0 ? eval_val(t.a,σ,ctx,aidx,nA) :                 # a==b
            (σ[idx]==1 ? eval_val(t.a,σ,ctx,aidx,nA) : eval_val(t.b,σ,ctx,aidx,nA))
    elseif t isa ThMax
        idx = ctx.thmin[objectid(t)]                            # sigma=1 iff a<b -> max=b
        idx == 0 ? eval_val(t.a,σ,ctx,aidx,nA) :
            (σ[idx]==1 ? eval_val(t.b,σ,ctx,aidx,nA) : eval_val(t.a,σ,ctx,aidx,nA))
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
# a / b  where b must be a denominator-free 1- or 2-term exp-polynomial with
# CONSTANT (rational) coefficients -- e.g. 1 - exp(-bL) (VMMC), 1 + exp(-bL)
# (Barker/Glauber), 2 - exp(-bL), or a single exp-monomial. The whole binomial
# factor b.num is recorded in the denominator multiset; the DB residual later
# clears it exactly. Soundness does not depend on b being sign-definite: the
# residual is multiplied through by the common denominator and the resulting
# exp-polynomial is tested for being identically zero, which (by continuity of the
# transition probabilities) is equivalent to detailed balance whatever the
# denominator -- a non-constant coefficient or >2 terms is still rejected because
# the engine only represents binomial denominators.
function val_div(a::Val, b::Val)
    isempty(b.den) || cant("division by a thresholded value that itself has a denominator")
    n = length(b.num)
    (1 <= n <= 2) || cant("division by a non-binomial threshold ($n exp-terms; only 1 or 2 supported)")
    for (_, p) in b.num
        ok, _ = poly_isconst(p)
        ok || cant("division by a threshold with a non-constant (polynomial) coefficient")
    end
    Val(a.num, vcat(a.den, [deepcopy(b.num)]))
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
    check_pairs    :: Vector{Int}              # pair indices actually checked (symmetry reps)
    sym_names      :: Vector{String}           # graph-verified symmetry generators used
    check_targets  :: Vector{Int}              # target STATES checked in -balance mode (one per orbit)
end

# ---- species-permutation (type) symmetry support ----------------------------
# A permutation of the species labels relabels the coupling atoms (it is a
# bijection of coupling space), so it is a symmetry of the detailed-balance problem
# whenever the algorithm is species-EQUIVARIANT -- the same logic as the spatial
# p4m reduction, but the action permutes the symbolic atoms rather than fixing
# them. It is admitted only after being verified on the computed graph (energy and
# transition weights match the atom-permuted originals), so nothing is assumed of
# the algorithm. See doc/type-taint.md / AUDIT for the analysis.

# Relabel a RatForm's coupling atoms by a species permutation.
_perm_rat(rf::RatForm, σ::Dict{Int,Int})::RatForm =
    RatForm((a.iscoupling ? Jc(σ[a.a], σ[a.b], a.d2) : a) => c for (a, c) in rf)
# Same, returning a LinForm (tau-free) for the th_* builders.
_perm_linform(rf::RatForm, σ::Dict{Int,Int})::LinForm =
    (lf = LinForm(); for (a, c) in rf; addcoef!(lf, a.iscoupling ? Jc(σ[a.a], σ[a.b], a.d2) : a, TauNum(c)); end; lf)

# Rebuild a threshold with its coupling atoms relabeled by σ, re-interning so the
# result is the SAME object the BFS produced for the σ-image (if any) -> objectid
# match. A genuinely new (never-seen) threshold just fails the later key lookup.
function permute_atoms_th(t::ThExpr, σ::Dict{Int,Int})::ThExpr
    if t isa ThConst; t
    elseif t isa ThBoltz;  th_boltz(_perm_linform(t.L, σ))
    elseif t isa ThLinear; th_linear(_perm_linform(t.L, σ))
    elseif t isa ThOp
        a = permute_atoms_th(t.a, σ); b = permute_atoms_th(t.b, σ)
        t.op === :+ ? th_add(a,b) : t.op === :- ? th_sub(a,b) :
        t.op === :* ? th_mul(a,b) : th_div(a,b)
    elseif t isa ThMin; th_min(permute_atoms_th(t.a,σ), permute_atoms_th(t.b,σ))
    elseif t isa ThMax; th_max(permute_atoms_th(t.a,σ), permute_atoms_th(t.b,σ))
    else  # ThPiece
        cls = Tuple{Vector{Cond},ThExpr}[
            (Cond[Cond(_perm_rat(g.lhs, σ), g.strict) for g in gs], permute_atoms_th(v, σ))
            for (gs, v) in t.clauses]
        th_piece(cls, permute_atoms_th(t.default, σ))
    end
end

# Generators of the multiplicity-preserving species-permutation group: adjacent
# transpositions within each set of labels that share a multiplicity. (Labels of
# different multiplicity can never be swapped without changing the state multiset.)
function _type_generators(typemult::Vector{Int})
    cnt = Dict{Int,Int}(); for t in typemult; cnt[t] = get(cnt,t,0) + 1; end
    bymult = Dict{Int,Vector{Int}}(); for (t,m) in cnt; push!(get!(bymult,m,Int[]), t); end
    gens = Tuple{String,Dict{Int,Int}}[]
    for (_, labels) in bymult
        ls = sort(labels)
        for k in 1:length(ls)-1
            σ = Dict(t => t for t in keys(cnt)); σ[ls[k]] = ls[k+1]; σ[ls[k+1]] = ls[k]
            push!(gens, ("type($(ls[k])<->$(ls[k+1]))", σ))
        end
    end
    gens
end

# Reduce the detailed-balance pair set using symmetries VERIFIED on the COMPUTED
# transition graph -- never assumed of the algorithm. For a candidate symmetry g
# (a state-index permutation, optionally carrying an atom relabeling), if (i) the
# energy is g-equivariant and (ii) the directed transition graph is g-equivariant
# -- every edge (s->d) maps to an edge (g.s -> g.d) whose weights are the atom-
# relabeled originals -- then for all couplings T(g.s->g.d) = g.T(s->d) and
# pi(g.s) = g.pi(s), so DB for (g.s, g.d) is equivalent to DB for (s,d) (the
# residual is the same up to the bijective atom relabeling). DB then need be
# evaluated on only one pair per orbit of the group generated by the verified g's.
#
# Candidates: all elements of the lattice point group p4m (translations + D4),
# whose atom action is the IDENTITY, AND the species-permutation generators, whose
# atom action is the corresponding atom relabeling. Any subgroup is detected.
#
# Soundness: for a spatial g the atom action is the identity, so equal weight-index
# multisets => identical interned Leaf objects => identical symbolic weights. For a
# species g the atom action relabels atoms, so we check that each edge maps to one
# whose weight indices are the ATOM-RELABELED originals (`wperm[wi]`), which is
# again a SUFFICIENT exact test (a matched key means the same interned objects, i.e.
# the relabel of the original weight). Energy is checked exactly (integer-vector,
# permuted for species g). A candidate that does not verify is simply dropped, so
# more pairs are checked, never fewer than correctness requires -- this can only
# speed the check, never alter the verdict. (Weight indices are canonical in serial
# runs; under -parallel, thread-local interning may split equal weights, in which
# case a real symmetry fails to verify and is conservatively skipped.)
function _verified_graph_symmetry_reps(bfs::BFSResult,
        energy_coeffs::Vector{Vector{Q}}, pairs::Vector{Tuple{Int,Int}},
        uweights::Vector{Leaf}, atoms::Vector{Atom}, aidx::Dict{Atom,Int})
    states = bfs.states; idx = bfs.idx; n = bfs.n; S = length(states); nA = length(atoms)
    ew = Dict{Tuple{Int,Int},Vector{Int}}()                  # directed edge -> weight indices
    for (s,d,wi) in bfs.trans; push!(get!(ew,(s,d),Int[]), wi); end
    for v in values(ew); sort!(v); end
    function permof(f)   # state-index permutation, or nothing if not a bijection
        π = Vector{Int}(undef, S)
        for i in 1:S
            j = get(idx, f(states[i]), 0); j == 0 && return nothing
            π[i] = j
        end
        length(Set(π)) == S ? π : nothing
    end
    nW = length(uweights)
    keymap = Dict{Any,Int}(); for wi in 1:nW; keymap[_weight_key(uweights[wi])] = wi; end

    # Candidate = (name, statePerm, atomIdxPerm | nothing, speciesPerm | nothing).
    # Spatial p4m elements fix the atoms (both nothing -> identity action). Species
    # generators relabel atoms: atomIdxPerm permutes the energy coefficient vector,
    # and the species permutation σ relabels each weight's atoms (its image weight
    # index is looked up lazily during the graph check, so a non-equivariant species
    # candidate bails on the first missing image rather than mapping every weight).
    cands = Tuple{String,Vector{Int},Union{Nothing,Vector{Int}},Union{Nothing,Dict{Int,Int}}}[]
    for (nm, f) in (("translate(1,0)", cs->translate_cstate(cs,1,0,n)),
                    ("translate(0,1)", cs->translate_cstate(cs,0,1,n)),
                    ("rotate90", cs->rotate_cstate(cs,n)), ("rotate180", cs->rotate180_cstate(cs,n)),
                    ("rotate270", cs->rotate270_cstate(cs,n)), ("reflect", cs->reflect_cstate(cs,n)),
                    ("reflect_h", cs->reflect_h_cstate(cs,n)), ("reflect_v", cs->reflect_v_cstate(cs,n)),
                    ("reflect_ad", cs->reflect_ad_cstate(cs,n)))
        π = permof(f); π === nothing && continue
        push!(cands, (nm, π, nothing, nothing))
    end
    typemult = Int[t for (r,c,t) in states[1]]
    for (nm, σ) in _type_generators(typemult)
        π = permof(cs -> sort(NTuple{3,Int}[(r, c, σ[t]) for (r,c,t) in cs])); π === nothing && continue
        atomperm = Vector{Int}(undef, nA); ok = true     # atom index permutation
        for (k, a) in enumerate(atoms)
            b = a.iscoupling ? Jc(σ[a.a], σ[a.b], a.d2) : a
            j = get(aidx, b, 0); j == 0 && (ok = false; break); atomperm[k] = j
        end
        ok && push!(cands, (nm, π, atomperm, σ))
    end

    verified = Vector{Int}[]; names = String[]
    for (nm, π, atomperm, σ) in cands
        # (i) energy g-equivariance
        eok = true
        if atomperm === nothing
            for i in 1:S; energy_coeffs[π[i]] == energy_coeffs[i] || (eok = false; break); end
        else
            for i in 1:S
                v = energy_coeffs[i]; o = zeros(Q, nA); for a in 1:nA; o[atomperm[a]] = v[a]; end
                energy_coeffs[π[i]] == o || (eok = false; break)
            end
        end
        eok || continue
        # (ii) graph g-equivariance. For a species candidate, wpix(wi) is the atom-
        # relabeled image weight index (0 = absent), computed on demand and memoised.
        wcache = Dict{Int,Int}()
        wpix(wi) = get!(wcache, wi) do
            lf = uweights[wi]
            pl = Leaf(lf.next, lf.coeff,
                      ThFactor[ThFactor(permute_atoms_th(f.thr, σ), f.accepted) for f in lf.factors])
            get(keymap, _weight_key(pl), 0)
        end
        gok = true
        for ((s,d), w) in ew
            w2 = get(ew, (π[s], π[d]), nothing)
            if w2 === nothing; gok = false; break; end
            if σ === nothing
                w2 == w || (gok = false; break)
            else
                pw = Int[]; bad = false
                for wi in w; x = wpix(wi); x == 0 && (bad = true; break); push!(pw, x); end
                (bad || (sort!(pw); pw != w2)) && (gok = false; break)
            end
        end
        gok && (push!(verified, π); push!(names, nm))
    end

    # Union-find over pair indices under the verified generators -> orbit reps.
    P = length(pairs)
    pidx = Dict{Tuple{Int,Int},Int}(); for (k,(i,j)) in enumerate(pairs); pidx[(i,j)] = k; end
    parent = collect(1:P)
    findset(x) = (while parent[x] != x; parent[x] = parent[parent[x]]; x = parent[x]; end; x)
    for k in 1:P
        (i,j) = pairs[k]
        for π in verified
            a,b = π[i], π[j]; kk = get(pidx, a < b ? (a,b) : (b,a), 0)
            if kk != 0; ra = findset(k); rb = findset(kk); ra != rb && (parent[ra] = rb); end
        end
    end
    (Int[k for k in 1:P if findset(k) == k], names, verified)
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
    thmin = Dict{UInt,Int}()                       # ThMin/ThMax objectid -> cond index (0 = a==b)
    function scan!(t::ThExpr)
        if t isa ThOp; scan!(t.a); scan!(t.b)
        elseif t isa ThMin || t isa ThMax          # both switch on the a<b hyperplane
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
        elseif t isa ThMin || t isa ThMax; (i=thmin[objectid(t)]; i!=0 && push!(acc,i)); factor_conds(t.a,acc); factor_conds(t.b,acc)
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

    # Symmetry reduction of the pair set, verified on the computed graph (sound;
    # speed only). Falls back to all pairs when no symmetry verifies.
    check_pairs, sym_names, state_perms = _verified_graph_symmetry_reps(bfs, energy_coeffs, pairs,
                                                           uweights, atoms, aidx)

    # State-orbit reps under the SAME verified group (for the -balance column check:
    # the balance residual B_{g·t} is identically zero iff B_t is, so one target per
    # orbit suffices — the column analogue of the pair-orbit reduction).
    nS = length(bfs.states); parentS = collect(1:nS)
    findS(x) = (while parentS[x] != x; parentS[x] = parentS[parentS[x]]; x = parentS[x]; end; x)
    for π in state_perms, s in 1:nS
        a = findS(s); b = findS(π[s]); a != b && (parentS[a] = b)
    end
    check_targets = Int[s for s in 1:nS if findS(s) == s]

    DBModel(aidx, nA, eff_list, strict_list, energy_coeffs, ctx,
            uweights, uw_active, pairs, ij_srcs, ji_srcs, check_pairs, sym_names, check_targets)
end

# ---- Phase 2: chamber enumeration via EXACT rational LP + degenerate filter ----
#
# Chamber feasibility is decided by an EXACT rational simplex, not a floating-
# point LP. This removes the only non-exact step the previous design had (a HiGHS
# LP with a 1e-6 feasibility tolerance), and with it the last theoretical route to
# a wrong verdict from numerical error. It also drops a heavy binary dependency,
# so the checker starts faster and runs in more environments. The exact result is
# validated to agree with HiGHS on every sign pattern of every bundled example.

# Exact two-phase primal simplex with Bland's rule (guaranteed termination, no
# cycling):  maximize c·x  s.t.  A x <= b,  x >= 0,  over the rationals.
# Returns (status, optimum) with status in (:optimal, :unbounded, :infeasible).
function simplex_max(A::Matrix{Q}, b::Vector{Q}, c::Vector{Q})
    m, n = size(A)
    needart = [b[i] < 0 for i in 1:m]; nart = count(needart)
    ncol = n + m + nart
    T = zeros(Q, m + 1, ncol + 1); basis = zeros(Int, m); ai = 0
    for i in 1:m
        s = needart[i] ? Q(-1) : Q(1)          # scale row so RHS >= 0
        for j in 1:n; T[i, j] = s * A[i, j]; end
        T[i, n + i] = s; T[i, ncol + 1] = s * b[i]
        if needart[i]; ai += 1; T[i, n + m + ai] = Q(1); basis[i] = n + m + ai
        else;          basis[i] = n + i; end
    end
    function pivot!(prow, pcol)
        T[prow, :] ./= T[prow, pcol]
        for r in 1:size(T, 1)
            r == prow && continue
            f = T[r, pcol]; f == 0 && continue
            T[r, :] .-= f .* T[prow, :]
        end
        basis[prow] = pcol
    end
    function optimize!(cols)
        while true
            pcol = 0
            for j in cols; if T[m + 1, j] > 0; pcol = j; break; end; end  # Bland
            pcol == 0 && return :optimal
            prow = 0; best = Q(0)
            for i in 1:m
                if T[i, pcol] > 0
                    r = T[i, ncol + 1] / T[i, pcol]
                    if prow == 0 || r < best || (r == best && basis[i] < basis[prow])
                        best = r; prow = i
                    end
                end
            end
            prow == 0 && return :unbounded
            pivot!(prow, pcol)
        end
    end
    if nart > 0                                  # Phase I: drive out artificials
        for j in (n + m + 1):ncol; T[m + 1, j] = Q(-1); end
        for i in 1:m; basis[i] > n + m && (T[m + 1, :] .+= T[i, :]); end
        optimize!(1:(n + m + nart))
        T[m + 1, ncol + 1] != 0 && return (:infeasible, Q(0))
        for i in 1:m                             # pivot any zero-valued artificial out
            if basis[i] > n + m
                pcol = 0
                for j in 1:(n + m); if T[i, j] != 0; pcol = j; break; end; end
                pcol != 0 && pivot!(i, pcol)
            end
        end
        for j in 1:(ncol + 1); T[m + 1, j] = Q(0); end
    end
    for j in 1:n; T[m + 1, j] = c[j]; end        # Phase II: maximize c·x
    for i in 1:m
        cb = basis[i] <= n ? c[basis[i]] : Q(0)
        cb == 0 && continue
        T[m + 1, :] .-= cb .* T[i, :]
    end
    optimize!(1:(n + m)) == :unbounded && return (:unbounded, Q(0))
    (:optimal, -T[m + 1, ncol + 1])
end

# Exact open-chamber feasibility for a sign pattern `sigma` over the homogeneous
# conditions (eff[i]·J, strict[i]). Substituting J = u - 1 (u in [0,2]) and
# maximizing a slack t pushed into every "strict-side" inequality, the chamber is
# a genuine open region iff the exact optimum t* > 0. (This is the eps->0+ limit
# of the old HiGHS test, so it agrees with it but with no tolerance.)
function _is_feasible(sigma::Vector{Int}, eff::Vector{Vector{Q}},
                      strict::Vector{Bool}, nA::Int)::Bool
    isempty(sigma) && return true
    k = length(sigma); nv = nA + 1; tcol = nA + 1
    rows = Vector{Q}[]; rhs = Q[]
    for i in 1:k
        s = sigma[i] == 1 ? Q(1) : Q(-1)
        strict_side = (sigma[i] == 1) == strict[i]
        row = zeros(Q, nv); sumeff = Q(0)
        for j in 1:nA; row[j] = -s * eff[i][j]; sumeff += eff[i][j]; end
        strict_side && (row[tcol] = Q(1))
        push!(rows, row); push!(rhs, -s * sumeff)
    end
    for j in 1:nA
        row = zeros(Q, nv); row[j] = Q(1); push!(rows, row); push!(rhs, Q(2))  # u_j <= 2
    end
    row = zeros(Q, nv); row[tcol] = Q(1); push!(rows, row); push!(rhs, Q(1))   # t <= 1
    A = permutedims(reduce(hcat, rows))
    c = zeros(Q, nv); c[tcol] = Q(1)
    st, z = simplex_max(A, rhs, c)
    st == :optimal && z > 0
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
    # Generic interior witness for an initial chamber: with J*_a = base^a the
    # highest-index term dominates, so no condition whose integer coefficients sum
    # to < base can vanish. base is chosen as large as Int128 allows for this nA
    # (100 for the small arrangements here; smaller, still > coefficient sums, when
    # nA is large) so the witness stays exact and overflow-free.
    base = 100
    while base > 4 && big(base)^nA > typemax(Int128); base -= 1; end
    Jstar = Q[Q(base)^a for a in 1:nA]
    initial = Int[ (sum(m.cond_eff_lhs[i][a] * Jstar[a] for a in 1:nA) >= 0) ? 1 : 0
                   for i in 1:k ]
    # The chamber-adjacency graph (chambers differing in one hyperplane's sign
    # share a facet) is connected, so this BFS reaches every chamber given an
    # EXACT feasibility test.
    visited = Set{Vector{Int}}([copy(initial)])
    feasible = [copy(initial)]; queue = [copy(initial)]
    while !isempty(queue)
        s = popfirst!(queue)
        for i in 1:k
            s2 = copy(s); s2[i] = 1 - s2[i]
            s2 in visited && continue
            push!(visited, copy(s2))
            if _is_feasible(s2, m.cond_eff_lhs, m.cond_is_strict, nA)
                push!(feasible, copy(s2)); push!(queue, copy(s2))
            end
        end
    end
    _filter_degenerate(feasible, m.cond_eff_lhs, m.cond_is_strict, k)
end

# ---- Phase 3: exact DB check (lazy Val cache + per-pair condition projection) ----
# Returns (pass, violations, nChambers); each violation is (i, j, chamber_index).
# The pair loop is independent per pair, so it is the natural parallel unit; each
# thread keeps its own Val cache (correctness is unaffected — the cache only
# memoises pure exact computations) and its own violation list.
function run_db_check(m::DBModel; parallel::Bool=false, use_symmetry::Bool=true,
                      mode::Symbol=:detailed)
    chambers = enumerate_chambers(m); nA = m.nA
    nt = Threads.maxthreadid()
    caches = [Dict{Tuple{Int,Vector{Int}}, Val}() for _ in 1:nt]
    # Lazy, cached leaf Val: depends only on the projection of sigma onto the
    # weight's active conditions, so one evaluation serves every chamber sharing
    # that projection.
    function leaf_val(wi::Int, σ::Vector{Int})
        cache = caches[Threads.threadid()]
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
        D = BSum[]
        for wi in m.ij_srcs[p]; D = ms_unionmax(D, leaf_val(wi,σ).den); end
        for wi in m.ji_srcs[p]; D = ms_unionmax(D, leaf_val(wi,σ).den); end
        res = BSum()
        for wi in m.ij_srcs[p]
            v = leaf_val(wi,σ)
            for (L,c) in bs_mul(bs_shift(v.num, ei), expand_binoms(ms_diff(D,v.den), nA)); bs_add!(res,L,c); end
        end
        for wi in m.ji_srcs[p]
            v = leaf_val(wi,σ)
            for (L,c) in bs_mul(bs_shift(v.num, ej), expand_binoms(ms_diff(D,v.den), nA)); bs_add!(res,L,poly_neg(c)); end
        end
        isempty(res)
    end

    # ---- GLOBAL BALANCE (mode=:balance) -------------------------------------
    # Stationarity pi*T = pi is the COLUMN sum of the detailed-balance residual
    # matrix: for each target state t,  B_t = sum_s [ pi_s T(s->t) - pi_t T(t->s) ] = 0
    # (the s==t term cancels, so only off-diagonal transitions enter — exactly what the
    # graph stores). Detailed balance is the STRONGER per-pair condition R_{st}=0; an
    # algorithm can satisfy balance (correct sampling) while violating DB (the entire
    # non-reversible family: event-chain, lifting, Suwa-Todo). Because the B_t sum mixes
    # pairs with different (1-exp) denominators, we aggregate the directed leaf
    # contributions over a COMMON denominator (not the per-pair-cleared numerators),
    # so the test stays exact. B_{g.t} is identically zero iff B_t is (same verified
    # group as the pair reduction), so one target per state-orbit suffices.
    if mode === :balance
        nS = length(m.energy_coeffs)
        incident = [Tuple{Int,Int}[] for _ in 1:nS]      # incident[t] = [(pair, +1 if t==j else -1)]
        for (p, (i,j)) in enumerate(m.pairs); push!(incident[j], (p, 1)); push!(incident[i], (p, -1)); end
        function balance_zero(t::Int, σ::Vector{Int})
            ws = Tuple{Int,Int,Int}[]                    # (weight-index, energy-state, sign)
            for (p, s) in incident[t]
                i, j = m.pairs[p]
                if s > 0      # t == j:  +T(i->t)pi_i  - T(t->i)pi_t
                    for wi in m.ij_srcs[p]; push!(ws, (wi, i, 1)); end
                    for wi in m.ji_srcs[p]; push!(ws, (wi, t, -1)); end
                else          # t == i:  +T(j->t)pi_j  - T(t->j)pi_t
                    for wi in m.ji_srcs[p]; push!(ws, (wi, j, 1)); end
                    for wi in m.ij_srcs[p]; push!(ws, (wi, t, -1)); end
                end
            end
            D = BSum[]
            for (wi,_,_) in ws; D = ms_unionmax(D, leaf_val(wi,σ).den); end
            res = BSum()
            for (wi, es, sgn) in ws
                v = leaf_val(wi,σ)
                term = bs_mul(bs_shift(v.num, m.energy_coeffs[es]), expand_binoms(ms_diff(D, v.den), nA))
                for (L,c) in term; bs_add!(res, L, sgn > 0 ? c : poly_neg(c)); end
            end
            isempty(res)
        end
        targets = use_symmetry ? m.check_targets : collect(1:nS)
        work_t  = Int[t for t in targets if !isempty(incident[t])]
        bviol_per = [Tuple{Int,Int,Int}[] for _ in 1:nt]
        function do_target(t::Int)
            apset = Set{Int}()
            for (p, _) in incident[t]
                for wi in m.ij_srcs[p]; union!(apset, m.uw_active[wi]); end
                for wi in m.ji_srcs[p]; union!(apset, m.uw_active[wi]); end
            end
            ap = sort(collect(apset)); seen = Set{Vector{Int}}(); out = bviol_per[Threads.threadid()]
            for (ridx, σ) in enumerate(chambers)
                proj = Int[σ[c] for c in ap]; proj in seen && continue; push!(seen, proj)
                balance_zero(t, σ) || push!(out, (t, t, ridx))
            end
        end
        if parallel && nt > 1
            Threads.@threads :static for t in work_t; do_target(t); end
        else
            for t in work_t; do_target(t); end
        end
        bviol = isempty(bviol_per) ? Tuple{Int,Int,Int}[] : reduce(vcat, bviol_per)
        return (isempty(bviol), bviol, length(chambers))
    end

    # Default: only one pair per graph-verified symmetry orbit (sound; faster).
    # use_symmetry=false checks every pair (the baseline the suite cross-checks against).
    scan_pairs = use_symmetry ? m.check_pairs : collect(eachindex(m.pairs))
    work = Int[p for p in scan_pairs
               if !(isempty(m.ij_srcs[p]) && isempty(m.ji_srcs[p]))]
    viol_per = [Tuple{Int,Int,Int}[] for _ in 1:nt]
    function do_pair(p::Int)
        # Active conditions for this pair: only these distinguish its chambers, so
        # check it once per distinct PROJECTION of the chambers onto them.
        apset = Set{Int}()
        for wi in m.ij_srcs[p]; union!(apset, m.uw_active[wi]); end
        for wi in m.ji_srcs[p]; union!(apset, m.uw_active[wi]); end
        ap = sort(collect(apset))
        seen = Set{Vector{Int}}(); out = viol_per[Threads.threadid()]
        for (ridx, σ) in enumerate(chambers)
            proj = Int[σ[c] for c in ap]
            proj in seen && continue
            push!(seen, proj)
            residual_zero(p, σ) || push!(out, (m.pairs[p][1], m.pairs[p][2], ridx))
        end
    end

    if parallel && nt > 1
        Threads.@threads :static for p in work; do_pair(p); end
    else
        for p in work; do_pair(p); end
    end
    violations = isempty(viol_per) ? Tuple{Int,Int,Int}[] : reduce(vcat, viol_per)
    (isempty(violations), violations, length(chambers))
end
