#!/usr/bin/env julia
# ============================================================================
# rotation_taint_poc.jl  —  can a "taint" trick bring ROTATION / REFLECTION
# into the tau-BFS, the way tau brings translation and the TypeTag brings
# species?
# ============================================================================
# Self-contained.  Run:  julia doc/rotation_taint_poc.jl
#
# Background (what already works in dbc.jl):
#   * tau certifies TRANSLATION-equivariance from a SINGLE symbolic BFS, because
#     a translation is a single continuous OFFSET applied to every particle; the
#     gate `is_covariant_pos` accepts an output position iff it is a pure
#     +offset shift (cr=1,cc=0 / cr=0,cc=1).  One BFS  ->  whole N^2 orbit.
#   * The TypeTag certifies SPECIES-equivariance from a single tagged BFS,
#     because a species relabel is used only equivariantly (==, hash, atom build)
#     so a clean BFS means control flow never saw an absolute label.  The logic
#     is "control flow is label-blind  =>  equivariant for all sigma".
#   * D4 (rot/refl, the lattice point group p4m) is used ONLY post-hoc in Step 4,
#     verified on the COMPUTED graph, never inside the BFS.
#
# Question: can rotation/reflection be certified DURING the BFS by a taint, to
# reduce the number of states BFS'd (the bottleneck) by up to |D4| = 8x?
#
# The experiments below test three candidate mechanisms:
#   A. symbolic single-BFS taint   (the tau-analogue)
#   B. control-flow-blind taint    (the species-analogue)
#   C. verify the D4 generators by re-BFS, compare leaf MULTISETS on the graph
#      (the sound "alternative 6.2" of type-taint.md, promoted to the BFS)
# ============================================================================

using Printf

const N = 4                                   # 4x4 torus
const Site  = Tuple{Int,Int}
const St    = Vector{NTuple{3,Int}}           # sorted (r,c,type)
canon(st)   = sort(st)

# ---- lattice point-group actions (match dbc.jl conventions) -----------------
rot90(r,c)  = (c, N+1-r)                       # 90 CCW
rot180(r,c) = (N+1-r, N+1-c)
rot270(r,c) = (N+1-c, r)
refl(r,c)   = (c, r)                           # main diagonal
applyg(g, st::St)::St = canon(NTuple{3,Int}[(g(r,c)..., t) for (r,c,t) in st])
applyg_leaves(g, lv::Dict{St,Rational{Int}}) =
    Dict{St,Rational{Int}}(applyg(g,s) => p for (s,p) in lv)

# energy: isotropic nearest-neighbour (function of pairwise minimum-image d2)
pbc_d2(a::Site,b::Site) = let dr=min(mod(a[1]-b[1],N), mod(b[1]-a[1],N)),
                              dc=min(mod(a[2]-b[2],N), mod(b[2]-a[2],N)); dr^2+dc^2 end
energy(st::St) = sum(pbc_d2((st[i][1],st[i][2]),(st[j][1],st[j][2]))==1 ? 1 : 0
                     for i in 1:length(st) for j in i+1:length(st); init=0)

# ============================================================================
# Algorithms as LEAF ENUMERATORS: state -> Dict(next_state => probability).
# (The real BFS enumerates every rng outcome into exactly such a multiset.)
# ============================================================================

# isotropic single/multi-particle hop, all 4 neighbours equal, blocked->stay.
function leaves_iso(st::St)::Dict{St,Rational{Int}}
    occ = Set{Site}((r,c) for (r,c,_) in st); K = length(st)
    out = Dict{St,Rational{Int}}(); p = 1//(4K)
    for i in 1:K
        (r,c,t) = st[i]
        for (dr,dc) in ((1,0),(-1,0),(0,1),(0,-1))
            nr=mod(r-1+dr,N)+1; nc=mod(c-1+dc,N)+1
            nst = (nr,nc) in occ ? canon(copy(st)) :
                  canon(NTuple{3,Int}[j==i ? (nr,nc,t) : st[j] for j in 1:K])
            out[nst] = get(out,nst,0//1) + p
        end
    end
    out
end

# drift: same hop but North (dr=-1) is twice as likely -> breaks ALL of D4.
function leaves_drift(st::St)::Dict{St,Rational{Int}}
    occ = Set{Site}((r,c) for (r,c,_) in st); K = length(st)
    out = Dict{St,Rational{Int}}(); base = 1//(5K)         # weights 2+1+1+1 = 5
    for i in 1:K
        (r,c,t) = st[i]
        for (dr,dc,w) in ((-1,0,2),(1,0,1),(0,1,1),(0,-1,1))
            nr=mod(r-1+dr,N)+1; nc=mod(c-1+dc,N)+1
            nst = (nr,nc) in occ ? canon(copy(st)) :
                  canon(NTuple{3,Int}[j==i ? (nr,nc,t) : st[j] for j in 1:K])
            out[nst] = get(out,nst,0//1) + base*w
        end
    end
    out
end

# absolute-position move: a particle may hop North ONLY from row 1 (else stays).
# Equivariant under nothing in D4 (uses absolute row).
function leaves_absolute(st::St)::Dict{St,Rational{Int}}
    occ = Set{Site}((r,c) for (r,c,_) in st); K = length(st)
    out = Dict{St,Rational{Int}}(); p = 1//K
    for i in 1:K
        (r,c,t) = st[i]
        if r == 1
            nr=mod(r-1-1,N)+1
            nst = (nr,c) in occ ? canon(copy(st)) :
                  canon(NTuple{3,Int}[j==i ? (nr,c,t) : st[j] for j in 1:K])
            out[nst] = get(out,nst,0//1) + p
        else
            out[canon(copy(st))] = get(out,canon(copy(st)),0//1) + p
        end
    end
    out
end

# CHIRAL plaquette rotation: pick a unit square uniformly among ALL N^2 of them
# (prob 1/N^2) and CW-cycle its contents.  The SET of all plaquettes is
# C4-invariant and CW->CW under rotation, so this is C4-equivariant; reflection
# maps CW->CCW, so it is NOT reflection-equivariant.  (Selecting the plaquette
# by a fixed corner of a particle would be orientation-dependent and break C4 --
# the set of all plaquettes is what makes this genuinely chiral-but-C4.)
function leaves_chiral(st::St)::Dict{St,Rational{Int}}
    occ = Dict{Site,Int}(); for (r,c,t) in st; occ[(r,c)] = t; end
    out = Dict{St,Rational{Int}}(); p = 1//(N*N)
    for r in 1:N, c in 1:N
        sq = Site[(r,c), (r, mod(c,N)+1), (mod(r,N)+1, mod(c,N)+1), (mod(r,N)+1, c)]
        newocc = copy(occ)                      # contents move corner k -> k+1 (CW)
        for k in 1:4
            src = sq[k]; dst = sq[mod(k,4)+1]
            haskey(occ, src) ? (newocc[dst] = occ[src]) : delete!(newocc, dst)
        end
        nst = canon(NTuple{3,Int}[(s[1],s[2],ty) for (s,ty) in newocc])
        out[nst] = get(out,nst,0//1) + p
    end
    out
end

# TRAP move: isotropic everywhere EXCEPT when the state occupies the special site
# (N,N), where it drifts North.  Equivariant at most states, NOT at states
# touching (N,N).  Used in E5 to expose the unsoundness of deriving an un-BFS'd
# state's leaves by rotation.
function leaves_trap(st::St)::Dict{St,Rational{Int}}
    any((r,c)==(N,N) for (r,c,_) in st) ? leaves_drift(st) : leaves_iso(st)
end

# ============================================================================
# E1 — point-group actions ARE torus automorphisms preserving d2 and energy.
# ============================================================================
function E1()
    println("E1 — rot/refl preserve squared distance & isotropic energy")
    sts = [St([(1,1,1),(2,3,1),(4,2,1)]), St([(1,2,1),(2,2,1),(3,4,1)])]
    ok = true
    for st in sts, g in (rot90,rot180,rot270,refl)
        ok &= energy(applyg(g,st)) == energy(st)
    end
    @printf("   energy g-invariant for all samples/elements : %s\n\n", ok)
end

# ============================================================================
# E2 — THE OBSTRUCTION.  A symbolic single-BFS taint (the tau-analogue) requires
# a per-rng-path correspondence: rng outcome on g.s must equal g.(outcome on s).
# For a direction-ENUMERATING move that fails, even though the move is rotation-
# equivariant.  So the tau-style certificate cannot see rotation.
# ============================================================================
function E2()
    println("E2 — per-rng-path correspondence FAILS under rotation (the obstruction)")
    st = St([(2,2,1)])                          # single free particle
    g  = rot90
    # outcome of the FIRST rng branch (move North = dr -1) on s, then rotate:
    north(state) = let (r,c,t)=state[1]; St([(mod(r-1-1,N)+1, c, t)]); end
    path_then_g = applyg(g, north(st))          # g.(rng=North on s)
    g_then_path = north(applyg(g, st))          # rng=North on (g.s)
    @printf("   g.(North . s)  = %s\n", path_then_g)
    @printf("   North .(g.s)   = %s\n", g_then_path)
    @printf("   identical per-path?  %s   <- a tau/species-style certificate needs YES\n",
            path_then_g == g_then_path)
    println("   => a single-BFS symbolic/control-flow taint DECLINES this equivariant move.\n")
end

# ============================================================================
# E3 — but the LEAF MULTISET is equivariant: leaves(g.s) == g.leaves(s).
# This is what the BFS actually compares, so a graph-level check certifies it.
# ============================================================================
function multiset_equivariant(leaves, st::St, g)
    leaves(applyg(g,st)) == applyg_leaves(g, leaves(st))
end
function E3()
    println("E3 — leaf MULTISET is equivariant for the isotropic move (Mechanism C basis)")
    sts = [St([(2,2,1)]), St([(1,1,1),(2,3,1)]), St([(1,1,1),(1,2,1),(3,3,1)])]
    for g in (rot90,rot180,rot270,refl)
        ok = all(multiset_equivariant(leaves_iso, st, g) for st in sts)
        @printf("   %-8s : leaves(g.s)==g.leaves(s) for all samples : %s\n", string(g), ok)
    end
    println()
end

# ============================================================================
# E4 — Mechanism C is TIGHT and DISCRIMINATING: it declines biased / absolute
# moves, and for the chiral move it ACCEPTS the C4 elements but DECLINES the
# reflections (correct subgroup discovery, exactly like Step-4's p4m search).
# ============================================================================
function E4()
    println("E4 — Mechanism C is tight & discovers the correct subgroup")
    # include ADJACENT pairs/triples sharing a unit plaquette, so the chiral
    # (CW) move is actually exercised (isolated particles hop isotropically).
    sts = [St([(2,2,1)]), St([(1,1,1),(2,3,1)]),
           St([(1,1,1),(1,2,1)]), St([(2,2,1),(2,3,1),(3,3,1)])]
    for (name, lv) in (("iso", leaves_iso), ("drift", leaves_drift),
                       ("absolute", leaves_absolute), ("chiral", leaves_chiral))
        verdict = String[]
        for (gname,g) in (("rot90",rot90),("rot180",rot180),("rot270",rot270),("refl",refl))
            all(multiset_equivariant(lv, st, g) for st in sts) && push!(verdict, gname)
        end
        @printf("   %-9s equivariant under: %s\n", name,
                isempty(verdict) ? "(none)" : join(verdict, ", "))
    end
    println("   (iso: all D4 | drift,absolute: none | chiral: the 3 rotations, NOT refl)\n")
end

# ============================================================================
# E5 — THE SOUNDNESS WALL.  Why no taint/generator trick can SKIP a state.
#
# tau & species save BFS because a SINGLE BFS of the rep certifies equivariance
# for EVERY group element AT THE REP, so each state is reached from its rep by
# ONE certified element -- no un-verified state is ever trusted.
#
# Rotation has no single-BFS certificate (E2).  The only way to "verify" it is to
# compare leaves(g.s) with g.leaves(s) -- which needs leaves(g.s), i.e. BFS-ing
# g.s.  If instead we VERIFY generators only at the rep and DERIVE the rest by
# composition, we trust equivariance at states we never executed.  This is
# UNSOUND: an algorithm equivariant at the rep can fail at a derived orbit member.
# ============================================================================
function E5()
    println("E5 — deriving an un-BFS'd state by rotation is UNSOUND (the wall)")
    rep = St([(1,1,1)])                          # single particle
    # generator check AT THE REP (rot90, refl), as a 'verify generators' scheme would:
    gen_ok = multiset_equivariant(leaves_trap, rep, rot90) &&
             multiset_equivariant(leaves_trap, rep, refl)
    @printf("   generator check at rep {(1,1)} passes : %s\n", gen_ok)
    # ... yet r180.rep = (N,N) is the trap site; deriving its leaves is WRONG:
    derived = applyg_leaves(rot180, leaves_trap(rep))   # what we'd record if we SKIP it
    truth   = leaves_trap(applyg(rot180, rep))          # what a real BFS of it gives
    @printf("   r180.rep = %s (the trap site)\n", applyg(rot180, rep))
    @printf("   derived == truth (would be sound) : %s\n", derived == truth)
    @printf("   => skipping r180.rep records WRONG transitions -> risk of false PASS.\n")
    @printf("   => the ONLY sound check BFSes r180.rep itself: no BFS saved.\n\n")
end

# ============================================================================
# E7 — the naively-hoped-for orbit reduction (what D4 *would* give if it were
# free like translation) vs the reality from E5: it is NOT free, so 0 saved.
# ============================================================================
function enumerate_states(K::Int)
    sites = [(r,c) for r in 1:N for c in 1:N]
    out = St[]
    function rec(start, chosen)
        if length(chosen)==K; push!(out, canon(NTuple{3,Int}[(s[1],s[2],1) for s in chosen])); return; end
        for i in start:length(sites); rec(i+1, vcat(chosen, [sites[i]])); end
    end
    rec(1, Site[]); unique(out)
end
translate(st::St, dr, dc)::St = canon(NTuple{3,Int}[(mod(r-1+dr,N)+1, mod(c-1+dc,N)+1, t) for (r,c,t) in st])
function trans_orbit_reps(states)
    repof = Dict{St,St}(); reps = St[]
    for st in states
        haskey(repof, st) && continue
        orb = St[translate(st,dr,dc) for dr in 0:N-1 for dc in 0:N-1]
        rep = minimum(orb); push!(reps, rep)
        for t in orb; repof[t] = rep; end
    end
    reps, repof
end
identity_g(r,c)=(r,c)
refl90(r,c)  = rot90(refl(r,c)...)
refl180(r,c) = rot180(refl(r,c)...)
refl270(r,c) = rot270(refl(r,c)...)
function E7(K::Int)
    states = enumerate_states(K)
    treps, repof = trans_orbit_reps(states)
    crep_of = Dict{St,St}(); creps = St[]
    for tr in treps
        haskey(crep_of, tr) && continue; push!(creps, tr)
        for g in (identity_g, rot90, rot180, rot270, refl, refl90, refl180, refl270)
            img = repof[applyg(g, tr)]
            haskey(crep_of, img) || (crep_of[img] = tr)
        end
    end
    @printf("E7 — K=%d on %dx%d torus : orbit counts\n", K, N, N)
    @printf("   states                              : %d\n", length(states))
    @printf("   translation reps        (BFS today) : %d\n", length(treps))
    @printf("   translation x D4 reps   (hoped-for) : %d\n", length(creps))
    @printf("   hoped-for naive factor              : %.2fx  (UNREACHABLE soundly -- see E5)\n",
            length(treps)/length(creps))
    @printf("   sound BFS reduction from D4         : 1.00x (none)\n\n")
end

# ============================================================================
# E6 — determinism: Mechanism C is a pure function of the enumerated leaves.
# ============================================================================
function E6()
    st = St([(1,1,1),(2,3,1)])
    runs = Set(multiset_equivariant(leaves_iso, st, rot90) for _ in 1:1000)
    @printf("E6 — determinism: distinct verdicts over 1000 runs = %d (want 1)\n\n", length(runs))
end

E1(); E2(); E3(); E4(); E5(); E6(); E7(2); E7(3)
