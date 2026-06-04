# ============================================================================
# Proof-of-concept for the "type-taint" certificate idea.
#
# Question: could a tau-analogue tag on the SPECIES LABEL certify, from a SINGLE
# BFS, that an algorithm is species-equivariant -- so the tau-BFS could be reduced
# over species orbits (not just translation orbits), speeding the bottleneck?
#
# We isolate the species question from the rest of the engine: a "state" is a list
# of (position, type); a mini-"algorithm" maps a state to leaves [(next, weight)]
# where weight is a multiset of coupling atoms (a,b,d). We test:
#   E1  the tag flags EXACTLY the non-equivariant uses of a type label
#   E2  SOUNDNESS: untainted  <=>  genuinely species-equivariant (ground truth by
#       brute-force sigma-relabel comparison)
#   E3  DETERMINISM: the taint verdict is reproducible
#   E4  FRAGILITY: which ordinary operations bypass or over-trip the tag
# ============================================================================

# ---- the type tag --------------------------------------------------------
const TAINT = Ref(false); const TMSG = Ref("")
taint!(m) = (TAINT[] = true; TMSG[] = m; nothing)
reset_taint!() = (TAINT[] = false; TMSG[] = "")

struct Tt; v::Int; end                         # a tagged species label

# ALLOWED (species-equivariant) uses: equality between two labels, hashing.
Base.:(==)(a::Tt, b::Tt) = a.v == b.v
Base.hash(a::Tt, h::UInt) = hash(a.v, h)
# FLAGGED uses: comparing to a literal, ordering, arithmetic -- these depend on the
# absolute label, not just the equality structure.
Base.:(==)(a::Tt, b::Integer) = (taint!("type == constant $b"); a.v == b)
Base.:(==)(a::Integer, b::Tt) = (taint!("constant $a == type"); a == b.v)
Base.isless(a::Tt, b::Tt)     = (taint!("type ordering (isless)"); a.v < b.v)
Base.isless(a::Tt, b::Integer)= (taint!("type < constant"); a.v < b)
Base.isless(a::Integer, b::Tt)= (taint!("constant < type"); a < b.v)
Base.:+(a::Tt, b)             = (taint!("type arithmetic +"); a.v + b)

# Jc is a TRUSTED equivariant primitive: it builds a coupling atom from two labels.
# Its internal canonicalisation (a<=b) is implementation detail, NOT a user branch,
# so it does not taint. It works on raw Ints or on tags (unwrapping internally).
_tv(x::Tt) = x.v; _tv(x::Integer) = x
Jc3(a, b, d) = (av = _tv(a); bv = _tv(b); av <= bv ? (av, bv, d) : (bv, av, d))

# ---- mini "states" and helpers ------------------------------------------
const L = 5                                    # ring of 5 sites
canon(state) = sort(state; by = x -> x[1])     # canonical order: by POSITION (type-free)
ringd(p, q) = min(mod(p - q, L), mod(q - p, L))
function energy_atoms(state)                    # multiset of Jc atoms over close pairs
    w = Tuple{Int,Int,Int}[]
    for i in 1:length(state), j in (i+1):length(state)
        d = ringd(state[i][1], state[j][1]); d == 1 && push!(w, Jc3(state[i][2], state[j][2], d))
    end
    sort(w)
end
relabel_state(state, σ) = [(p, σ[t isa Tt ? t.v : t]) for (p, t) in state]
relabel_atoms(w, σ) = sort([(min(σ[a], σ[b]), max(σ[a], σ[b]), d) for (a, b, d) in w])

# ---- mini "algorithms" (parametric in the label type) -------------------
# Each returns a Vector of leaves (next_state_canon, weight_atoms).

# (A) swap the types of the first adjacent distinct-type pair  -- EQUIVARIANT
function alg_swap(state)
    for i in 1:length(state)-1
        (p1,t1) = state[i]; (p2,t2) = state[i+1]
        if t1 != t2                                  # equality use only
            ns = copy(state); ns[i] = (p1, t2); ns[i+1] = (p2, t1)
            return [(canon(ns), energy_atoms(ns))]
        end
    end
    [(canon(state), energy_atoms(state))]
end

# (B) move the particle whose type == 1                       -- NOT equivariant
function alg_move_type1(state)
    for i in 1:length(state)
        (p,t) = state[i]
        if t == 1                                    # absolute-label branch
            ns = copy(state); ns[i] = (mod(p, L) + 1, t)
            return [(canon(ns), energy_atoms(ns))]
        end
    end
    [(canon(state), energy_atoms(state))]
end

# (C) pick the smallest-type particle and move it             -- NOT equivariant
function alg_min_type(state)
    srt = sort(state; by = x -> x[2])                # orders by type label
    (p,t) = srt[1]; i = findfirst(==(srt[1]), state)
    ns = copy(state); ns[i] = (mod(p, L) + 1, t)
    [(canon(ns), energy_atoms(ns))]
end

# (D) pure energy reweight (no move)                          -- EQUIVARIANT
alg_energy(state) = [(canon(state), energy_atoms(state))]

# (E) sort by (position, type) where positions are DISTINCT, then move first.
#     The type tie-break is never decisive => EQUIVARIANT, but the tag still trips
#     on isless(type).  This is the VMMC-style conservative false-positive.
function alg_tiebreak(state)
    srt = sort(state; by = x -> (x[1], x[2]))        # position first (always distinct)
    (p,t) = srt[1]; i = findfirst(==(srt[1]), state)
    ns = copy(state); ns[i] = (mod(p, L) + 1, t)
    [(canon(ns), energy_atoms(ns))]
end

# (F) UNWRAP escape hatch: read t.v directly and branch on it -- NOT equivariant,
#     but the tag is BYPASSED (no flag) -> demonstrates the discipline requirement.
function alg_unwrap(state)
    for i in 1:length(state)
        (p,t) = state[i]
        if (t isa Tt ? t.v : t) == 1                 # raw-value branch, dodges ==(Tt,Int)
            ns = copy(state); ns[i] = (mod(p, L) + 1, t)
            return [(canon(ns), energy_atoms(ns))]
        end
    end
    [(canon(state), energy_atoms(state))]
end

# ---- ground-truth equivariance (brute force over a sigma) ----------------
leaves_set(alg, state) = Set((nx, w) for (nx, w) in alg(state))
σ_apply(leaves, σ) = Set((canon(relabel_state(nx, σ)), relabel_atoms(w, σ)) for (nx, w) in leaves)
function is_equivariant(alg, state_int, σ)
    L_s   = alg(state_int)
    L_σs  = leaves_set(alg, relabel_state(state_int, σ))
    σL_s  = σ_apply(L_s, σ)
    L_σs == σL_s
end
function is_tainted(alg, state_int)
    reset_taint!()
    alg([(p, Tt(t)) for (p, t) in state_int])         # run with tagged labels
    TAINT[]
end

# ============================================================================
println("="^70)
println("E1+E2: taint verdict vs ground-truth species-equivariance")
println("="^70)
state = [(1,1), (2,2), (3,3)]                          # 3 species on a 5-ring
σ = Dict(1=>2, 2=>3, 3=>1)                              # a 3-cycle relabel
algs = [("alg_swap (equality only)", alg_swap),
        ("alg_move_type1 (==const)", alg_move_type1),
        ("alg_min_type (sort by type)", alg_min_type),
        ("alg_energy (Jc only)", alg_energy),
        ("alg_tiebreak (pos,then type)", alg_tiebreak),
        ("alg_unwrap (reads t.v)", alg_unwrap)]
for (nm, alg) in algs
    eq = is_equivariant(alg, state, σ)
    tn = is_tainted(alg, state)
    # SOUND if: tainted whenever NOT equivariant (no false "clean" on a broken alg).
    sound = eq || tn          # not-equivariant => must be tainted
    conservative = tn && eq   # tainted but actually equivariant (a harmless decline)
    verdict = !sound ? "*** UNSOUND (clean but not equivariant) ***" :
              conservative ? "conservative (flags an equivariant alg)" : "tight"
    println(rpad(nm, 32), " equivariant=", eq, "  tainted=", tn, "   -> ", verdict)
end

println("\n", "="^70); println("E3: determinism (taint verdict over 1000 repeats)"); println("="^70)
for (nm, alg) in algs
    vs = Set(is_tainted(alg, state) for _ in 1:1000)
    println(rpad(nm, 32), " distinct verdicts over 1000 runs = ", length(vs), "  (", collect(vs), ")")
end

println("\n", "="^70); println("E4: fragility — do ordinary ops trip / bypass the tag?"); println("="^70)
probe(desc, f) = (reset_taint!(); local r; try; r = f(); catch e; r = "ERROR:$(typeof(e))"; end;
                  println(rpad(desc, 40), " taint=", TAINT[], "  result=", r))
t1 = Tt(1); t2 = Tt(2)
probe("t1 == t2 (equality)",            () -> t1 == t2)
probe("t1 != t2",                       () -> t1 != t2)
probe("t1 == 1 (const)",                () -> t1 == 1)
probe("t1 in Set([t1,t2]) (hash/==)",   () -> t1 in Set([t1, t2]))
probe("t1 in [1,2,3] (Int vector)",     () -> t1 in [1, 2, 3])
probe("sort([t2,t1]) (isless)",         () -> sort([t2, t1]))
probe("unique([t1,t1,t2])",             () -> length(unique([t1, t1, t2])))
probe("t1.v  (unwrap escape hatch)",    () -> t1.v)
probe("Dict(t1=>10)[t1] (hash/==)",     () -> Dict(t1 => 10)[t1])
probe("string(t1)",                     () -> string(t1))
println("\n", "="^70)
println("E5: the taint is DYNAMIC — a tie-break taints only when it is DECISIVE")
println("="^70)
# Sort by (parity(pos), type): parity collides, so the type tie-break IS reached.
function alg_tiebreak_decisive(state)
    srt = sort(state; by = x -> (mod(x[1], 2), x[2]))
    (p,t) = srt[1]; i = findfirst(==(srt[1]), state)
    ns = copy(state); ns[i] = (mod(p, L) + 1, t)
    [(canon(ns), energy_atoms(ns))]
end
for (nm, alg) in [("tiebreak on (pos, type)   [pos distinct]", alg_tiebreak),
                  ("tiebreak on (parity, type)[parity collides]", alg_tiebreak_decisive)]
    println(rpad(nm, 44), " equivariant=", is_equivariant(alg, state, σ),
            "  tainted=", is_tainted(alg, state))
end
println("=> the tag fires per-run on type ops that ACTUALLY execute, not statically.")
println("\nDONE")
