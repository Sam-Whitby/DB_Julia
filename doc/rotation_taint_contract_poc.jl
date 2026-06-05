#!/usr/bin/env julia
# ============================================================================
# rotation_taint_contract_poc.jl  —  does SUPPLYING directions (a contract)
# restore single-BFS rotation certifiability?
# ============================================================================
# rotation_taint_poc.jl proved that under the CURRENT contract (directions are
# hardcoded constants inside the algorithm) there is no single-BFS rotation
# certificate: a hardcoded offset (1,0) is an ABSOLUTE quantity, so the per-rng
# path "move (1,0)" on s does not correspond to the same path on R.s (E2), and
# deriving an un-BFS'd state by rotation is unsound (E5).
#
# This PoC tests the user's proposal: make the direction set D a SUPPLIED,
# declared object (a params-file contract), require moves to be expressed as
# `pos + d` with d in D, and apply the rotation-taint to BOTH positions AND
# directions.  Claim to test:
#
#   If (i) D is declared CLOSED under the point group (a STATIC check on params),
#      (ii) a single BFS of the rep is CLEAN under a rotation-taint that flags any
#           non-covariant use of a position or direction (absolute coordinate,
#           direction-identity branch, anisotropic weighting),
#   then deriving leaves(R.s) := R.leaves(s) is SOUND for every state, with NO
#   execution of R.s -- exactly the way tau/species certify their orbits.
#
# We check: clean  <=>  (R.leaves(rep) == leaves(R.rep) for all R, all states),
# i.e. the taint accepts EXACTLY the soundly-derivable algorithms, and the E5
# trap that broke the old approach is now FLAGGED.
# ============================================================================

using Printf
const N = 4
const St = Vector{NTuple{3,Int}}
canon(x)::St = sort(x)
rot90(r,c)=(c,N+1-r); rot180(r,c)=(N+1-r,N+1-c); rot270(r,c)=(N+1-c,r); refl(r,c)=(c,r)
const GROUP = [("rot90",rot90),("rot180",rot180),("rot270",rot270),("refl",refl)]
applyg(g,st::St)::St = canon(NTuple{3,Int}[(g(r,c)..., t) for (r,c,t) in st])
applyg_leaves(g,lv)  = Dict{St,Rational{Int}}(applyg(g,s)=>p for (s,p) in lv)

# ---- the SUPPLIED direction set (the contract) and its STATIC closure check ----
const D = [(-1,0),(1,0),(0,1),(0,-1)]              # 4 cardinals, supplied to algos
# linear parts (direction transforms) of each group element
lin = Dict("rot90"=>((dr,dc)->(dc,-dr)), "rot180"=>((dr,dc)->(-dr,-dc)),
           "rot270"=>((dr,dc)->(-dc,dr)), "refl"=>((dr,dc)->(dc,dr)))
function D_closed_under(name)
    f = lin[name]; S = Set(D)
    all(f(d...) in S for d in D)
end

# ============================================================================
# The rotation-taint probe.  Algorithms are written against THIS API only.
# Covariant primitives never taint; any absolute use sets probe.tainted.
# ============================================================================
mutable struct Probe; tainted::Bool; end
Probe() = Probe(false)
# covariant primitives (clean):
mv(::Probe, pos, d)      = (mod(pos[1]-1+d[1],N)+1, mod(pos[2]-1+d[2],N)+1)  # pos + d
function dist2(::Probe, a, b)
    dr=min(mod(a[1]-b[1],N),mod(b[1]-a[1],N)); dc=min(mod(a[2]-b[2],N),mod(b[2]-a[2],N)); dr^2+dc^2
end
occ_has(::Probe, occ, pos) = pos in occ                  # set membership (covariant)
# NON-covariant primitives (taint): absolute coordinate / direction identity
rowof(p::Probe, pos) = (p.tainted = true; pos[1])
colof(p::Probe, pos) = (p.tainted = true; pos[2])
dir_is(p::Probe, d, ref) = (p.tainted = true; d == ref) # branch on absolute direction

# ============================================================================
# Algorithms, written ONLY against the supplied D and the probe API.
# ============================================================================
function alg_iso(pr::Probe, st::St)                     # isotropic: clean, full D4
    occ = Set((r,c) for (r,c,_) in st); K=length(st)
    out = Dict{St,Rational{Int}}(); w = 1//(K*length(D))
    for i in 1:K
        (r,c,t)=st[i]
        for d in D
            np = mv(pr,(r,c),d)
            nst = occ_has(pr,occ,np) ? canon(copy(st)) :
                  canon(NTuple{3,Int}[j==i ? (np[1],np[2],t) : st[j] for j in 1:K])
            out[nst] = get(out,nst,0//1)+w
        end
    end
    out
end
function alg_drift(pr::Probe, st::St)                    # weights North double: TAINTS (dir_is)
    occ = Set((r,c) for (r,c,_) in st); K=length(st)
    out = Dict{St,Rational{Int}}(); base = 1//(K*5)
    for i in 1:K
        (r,c,t)=st[i]
        for d in D
            wt = dir_is(pr,d,(-1,0)) ? 2 : 1            # <-- absolute direction identity
            np = mv(pr,(r,c),d)
            nst = occ_has(pr,occ,np) ? canon(copy(st)) :
                  canon(NTuple{3,Int}[j==i ? (np[1],np[2],t) : st[j] for j in 1:K])
            out[nst] = get(out,nst,0//1)+base*wt
        end
    end
    out
end
function alg_rowonly(pr::Probe, st::St)                  # moves only along rows: TAINTS
    occ = Set((r,c) for (r,c,_) in st); K=length(st)
    out = Dict{St,Rational{Int}}(); w = 1//(K*2)
    for i in 1:K
        (r,c,t)=st[i]
        for d in D
            dir_is(pr,d,(0,1)) || dir_is(pr,d,(0,-1)) || continue  # absolute axis choice
            np = mv(pr,(r,c),d)
            nst = occ_has(pr,occ,np) ? canon(copy(st)) :
                  canon(NTuple{3,Int}[j==i ? (np[1],np[2],t) : st[j] for j in 1:K])
            out[nst] = get(out,nst,0//1)+w
        end
    end
    out
end
function alg_trap(pr::Probe, st::St)                     # the E5 trap: TAINTS (rowof/colof)
    trapped = any(rowof(pr,(r,c))==N && colof(pr,(r,c))==N for (r,c,_) in st)
    trapped ? alg_drift(pr,st) : alg_iso(pr,st)
end

# ============================================================================
# Tests
# ============================================================================
const REPS = [St([(1,1,1)]), St([(2,2,1)]),
              St([(1,1,1),(2,3,1)]), St([(1,1,1),(1,2,1)]),
              St([(2,2,1),(2,3,1),(3,3,1)]),
              St([(4,4,1)]), St([(3,4,1),(4,4,1)])]      # incl. the trap site (4,4)

leaves(alg, st) = alg(Probe(), st)                       # leaves only (fresh probe)
function is_clean(alg, sts)                              # clean over a sample of states
    pr = Probe(); for st in sts; alg(pr, st); end; !pr.tainted
end
# derivation soundness: does deriving R.s := R.leaves(s) match a direct BFS of R.s?
function derivation_sound(alg, sts)
    for st in sts, (_,g) in GROUP
        applyg_leaves(g, leaves(alg,st)) == leaves(alg, applyg(g,st)) || return false
    end
    true
end

function main()
    println("C0 — STATIC: is the supplied direction set D closed under the point group?")
    for (name,_) in GROUP
        @printf("   D closed under %-7s : %s\n", name, D_closed_under(name))
    end
    println("   => closure is a property of the PARAMS, provable without running R.s\n")

    println("C1 — taint verdict  vs  derivation soundness  (should be identical columns)")
    @printf("   %-10s %-8s %-12s %-s\n", "algorithm", "clean?", "derivable?", "match?")
    for (name, alg) in (("iso",alg_iso),("drift",alg_drift),
                        ("rowonly",alg_rowonly),("trap",alg_trap))
        cl = is_clean(alg, REPS); dv = derivation_sound(alg, REPS)
        @printf("   %-10s %-8s %-12s %-s\n", name, cl, dv, cl==dv ? "OK" : "*** MISMATCH ***")
    end
    println("   => clean <=> soundly-derivable : the taint accepts EXACTLY the right algos\n")

    println("C2 — the E5 TRAP, revisited under the contract")
    trap_clean = is_clean(alg_trap, REPS)
    rep = St([(1,1,1)]); derived = applyg_leaves(rot180, leaves(alg_trap,rep))
    truth = leaves(alg_trap, applyg(rot180,rep))
    @printf("   trap clean under rotation-taint : %s  (was the silent unsound case)\n", trap_clean)
    @printf("   r180.rep=%s  derived==truth : %s\n", applyg(rot180,rep), derived==truth)
    @printf("   => taint FLAGS the trap -> it is DECLINED, never derived -> sound.\n\n")

    println("C3 — the clean ISO algorithm: single-BFS derivation reproduces direct BFS")
    ok = true
    for st in REPS, (gname,g) in GROUP
        d = applyg_leaves(g, leaves(alg_iso,st)); t = leaves(alg_iso, applyg(g,st))
        ok &= (d==t)
    end
    @printf("   R.leaves(rep) == leaves(R.rep) for all reps, all R in D4 : %s\n", ok)
    @printf("   (NO execution of R.rep needed -- certified from rep's own BFS + D-closure)\n\n")

    println("C4 — anisotropic D: if D is NOT closed, closure check FAILS (declines safely)")
    Dbad = [(-1,0),(1,0)]                                 # rows only
    badclosed = let f=lin["rot90"], S=Set(Dbad); all(f(d...) in S for d in Dbad); end
    @printf("   D=%s closed under rot90 : %s  -> engine would NOT offer rotation reduction\n",
            Dbad, badclosed)

    println("\nC5 — determinism")
    runs = Set(derivation_sound(alg_iso, REPS) for _ in 1:500)
    @printf("   distinct verdicts over 500 runs : %d (want 1)\n", length(runs))
end
main()
