# ============================================================================
# test_db.jl  —  Regression + stress suite for DB_Julia
# ============================================================================
# Three layers:
#   1. UNIT      — the building blocks (TauNum tracking, bit-exact selection
#                  weights, the exact rational-function engine, the exact LP).
#   2. FAIL-LOUD — every "I cannot do this exactly" path must raise CantHandle
#                  (or flag a tau violation), never silently guess.
#   3. EXAMPLES  — all bundled algorithms run end-to-end with their KNOWN
#                  (translational, detailed-balance, ergodicity) verdicts, so a
#                  false PASS OR a false FAIL anywhere fails the suite.
#
# Run:  julia --project=. test_db.jl          (serial)
#       julia --project=. -t auto test_db.jl  (also exercises the parallel path)
# ============================================================================

include(joinpath(@__DIR__, "dbc.jl"))
using Test, Random

seed_state(types, n) = (st = sort(types); pos = [(r,c) for r in 1:n for c in 1:n];
    sort(NTuple{3,Int}[(pos[k][1], pos[k][2], st[k]) for k in 1:length(st)]))

@testset "DB_Julia" begin

# ----------------------------------------------------------------------------
@testset "UNIT: TauNum linear tracking + nonlinear taint" begin
    a = tau_r_aug(3); b = tau_r_aug(1)
    @test is_tau_free(a - b) && tau0(a - b) == 2
    @test !is_tau_free(a^2) && tau0(a^2) == 9
    @test !is_tau_free(a + 5) && tau0(a + 5) == 8
    @test is_tau_free((a - b) * tau_const(4))
    @test !is_tau_free(tau_r_aug(1) * tau_c_aug(1))
    # covariance predicate: a unit-shifted position is covariant; a reflected or
    # absolute one is not.
    @test is_covariant_pos(Particle(tau_r_aug(1) + 2, tau_c_aug(1) - 1, 1))
    @test !is_covariant_pos(Particle(-tau_r_aug(1), tau_c_aug(1), 1))      # reflection
    @test !is_covariant_pos(Particle(TauNum(2), tau_c_aug(1), 1))          # absolute row
end

@testset "UNIT: position comparisons that would leak tau are hard errors" begin
    @test_throws CantHandle (tau_r_aug(1) == TauNum(1))     # absolute vs covariant
    @test_throws CantHandle (tau_r_aug(1) < tau_r_aug(2))   # ordering positions
    @test (tau_r_aug(2) == tau_r_aug(2)) == true            # same offset: allowed
    @test (TauNum(3) == TauNum(3)) == true                  # both tau-free: allowed
end

@testset "UNIT: nbits matches IntegerLength[n-1,2]" begin
    @test [nbits(n) for n in 1:9] == [0,1,2,2,3,3,3,3,4]
end

@testset "UNIT: rejection-sampling weights are exact (1/n), n leaves" begin
    function choice_weights(n)
        total = Q(0); leaves = 0; queue = [Int[]]
        while !isempty(queue)
            bits = popfirst!(queue); rng = BitSeqRNG(bits)
            try
                rand_choice_index!(rng, n); total += rng.coeff; leaves += 1
            catch e
                e isa OutOfBitsException ? (push!(queue,[bits;0]); push!(queue,[bits;1])) :
                e isa OutOfRangeException ? nothing : rethrow(e)
            end
        end
        (leaves, total)
    end
    for n in 1:9; l,w = choice_weights(n); @test l == n && w == 1; end
end

@testset "UNIT: Val engine — VMMC ratio cancels exactly" begin
    nA = 2; Lfwd = Q[1,0]; Lrev = Q[0,1]
    wRev  = val_oneminus(val_boltz(Lrev, nA), nA)
    wFwd  = val_oneminus(val_boltz(Lfwd, nA), nA)
    ratio = Val(wRev.num, [binom_oneminus(Lfwd, nA)])
    link  = val_mul(wFwd, ratio)
    @test bs_mul(link.num, expand_binoms(wRev.den, nA)) == bs_mul(wRev.num, expand_binoms(link.den, nA))
    D = ms_unionmax(link.den, wRev.den)
    res = bs_sub(bs_mul(link.num, expand_binoms(ms_diff(D,link.den),nA)),
                 bs_mul(wRev.num, expand_binoms(ms_diff(D,wRev.den),nA)))
    @test isempty(res)
end

@testset "UNIT: exact simplex LP" begin
    @test simplex_max(Q[1 0; 0 1; 1 1], Q[3,4,5], Q[1,1])    == (:optimal, Q(5))
    @test simplex_max(Q[1 1; 1 3], Q[4,6], Q[3,2])           == (:optimal, Q(12))
    @test simplex_max(Q[-1 0; 1 0], Q[-2,5], Q[1,0])         == (:optimal, Q(5))    # needs phase I
    @test simplex_max(Q[-1 0; 1 0], Q[-3,1], Q[1,0])[1]      == :infeasible
    @test simplex_max(reshape(Q[-1],1,1), Q[0], Q[1])[1]     == :unbounded
    @test simplex_max(Q[0 2; 0 1], Q[3,10], Q[0,1])          == (:optimal, Q(3)//2)
    # open-chamber feasibility: J1>0,J2>0 is open; J1>0 & J1<0 is not.
    @test  _is_feasible([1,1], [Q[1,0], Q[0,1]], [true,true], 2)
    @test !_is_feasible([1,0], [Q[1,0], Q[1,0]], [true,true], 2)
end

DISPS8 = [(dx,dy) for dx in -1:1 for dy in -1:1 if (dx,dy)!=(0,0)]
function mk_energy(n, maxd2)
    (state::PState) -> begin
        lf = LinForm()
        for i in 1:length(state), j in (i+1):length(state)
            d2 = pbc_d2(state[i], state[j], n); (0 < d2 <= maxd2) || continue
            addcoef!(lf, Jc(state[i].t, state[j].t, d2), TauNum(1))
        end
        lf
    end
end
function mk_metropolis(n, maxd2, energy)
    (rng, state::PState) -> begin
        pidx = rand_choice_index!(rng, length(state)); p = state[pidx]
        (dr,dc) = rand_choice!(rng, DISPS8); newp = Particle(p.r+dr, p.c+dc, p.t)
        rest = state[setdiff(1:length(state), pidx)]
        for q in rest; same_site(q, newp, n) && return state; end
        ns = vcat(rest, [newp]); dE = linsub(energy(ns), energy(state))
        metropolis!(rng, dE) ? ns : state
    end
end

@testset "UNIT: leaf weights sum to 1 at random coupling points" begin
    n=3; energy = mk_energy(n,2); step = mk_metropolis(n,2,energy)
    _init_threadlocal!()
    leaves = build_state_leaves(step, augmented_pstate(sort([(1,1,1),(2,2,2),(3,3,3)])), n, 22)
    @test !_tau_any()
    atoms = Set{Atom}(); for l in leaves, f in l.factors; _collect_atoms_th!(atoms, f.thr); end
    Random.seed!(1)
    for _ in 1:5
        J = Dict{Atom,Float64}(a => rand()*2-1 for a in atoms)
        @test abs(sum(eval_leaf(l, J, 1.0) for l in leaves) - 1.0) < 1e-9
    end
end

# ----------------------------------------------------------------------------
@testset "FAIL-LOUD: unsupported / dangerous inputs raise, never guess" begin
    n = 2
    # tau detection on an absolute-field energy: must flag, plain pairwise must not.
    energy_qf = function(state::PState)
        lf = mk_energy(n,2)(state)
        fld = TauNum(0); for p in state; fld = fld + p.r^2; end
        addcoef!(lf, Xparam(:fieldH), fld); lf
    end
    seed = augmented_pstate(sort([(1,1,1),(1,2,2)]))
    _init_threadlocal!(); build_state_leaves(mk_metropolis(n,2,energy_qf), seed, n, 22); @test _tau_any()
    _init_threadlocal!(); build_state_leaves(mk_metropolis(n,2,mk_energy(n,2)), seed, n, 22); @test !_tau_any()

    # a non-covariant (reflecting) move must flag a tau violation.
    reflect = (rng, st::PState) -> begin
        i = rand_choice_index!(rng, length(st)); p = st[i]
        vcat(st[setdiff(1:length(st), i)], [Particle(-p.r, p.c, p.t)])
    end
    _init_threadlocal!(); build_state_leaves(reflect, seed, n, 22); @test _tau_any()

    # maxdepth too small -> CantHandle (incomplete tree is fatal, not silent).
    @test_throws CantHandle build_state_leaves(mk_metropolis(3,2,mk_energy(3,2)),
                                augmented_pstate(sort([(1,1,1),(2,2,2),(3,3,3)])), 3, 1)
    # empty choice and inverted integer range -> CantHandle.
    @test_throws CantHandle rand_choice_index!(BitSeqRNG([0,1]), 0)
    @test_throws CantHandle rand_integer!(BitSeqRNG([0,1]), 5, 2)
    # ordering a position inside an algorithm -> CantHandle.
    badcmp = (rng, st::PState) -> (st[1].r < st[2].r ? st : st)
    @test_throws CantHandle build_state_leaves(badcmp, seed, n, 5)
end

# ----------------------------------------------------------------------------
@testset "PARALLEL: identical result to serial" begin
    n = 3; energy = mk_energy(n, 2); step = mk_metropolis(n, 2, energy)
    states = enumerate_states([1,2,3], n)
    bs = build_transitions(step, energy, states, n, 22; parallel=false)
    bp = build_transitions(step, energy, states, n, 22; parallel=true)
    @test bs.tau_free == bp.tau_free
    ps, _, cs = run_db_check(build_dbmodel(bs, energy); parallel=false)
    pp, _, cp = run_db_check(build_dbmodel(bp, energy); parallel=true)
    @test ps == pp && cs == cp           # same verdict, same chamber count
    # the transition graph is identical regardless of thread layout
    edges(b) = Set((s,d) for (s,d,_) in b.trans)
    @test edges(bs) == edges(bp)
end

# ----------------------------------------------------------------------------
# Full end-to-end verdicts. (tau, DB, ergodic) for every bundled example.
EXPECT = [
    ("single_metropolis",          (true,  true,  true )),
    ("kawasaki",                    (true,  true,  false)),   # ergodic FAIL by design
    ("quadratic_field",             (false, true,  true )),
    ("broken_variable_pool",        (true,  false, true )),
    ("broken_8way_hop",             (true,  false, true )),
    ("broken_biased_direction",     (true,  false, true )),
    ("broken_metropolis_halfbeta",  (true,  false, true )),
    ("broken_field_wrong_accept",   (true,  false, true )),
    ("vmmc_2d",                     (true,  true,  true )),
    ("hop_8way_correct",            (true,  true,  true )),   # power-of-two pool, 4x4
    ("metropolis_4x4",              (true,  true,  true )),   # larger lattice
    ("reflect_move",                (false, true,  false)),   # covariance guard -> fallback
    ("horizontal_metropolis",       (true,  true,  false)),   # D2 not D4; erg FAIL by design
    ("barker_accept",               (true,  true,  true )),   # (1+exp) denominator (Tier 1)
    ("poly_rate_accept",            (true,  true,  true )),   # polynomial weight factor (Tier 2)
    ("vmmc_2d_shuffle",             (true,  true,  true )),   # random candidate order -> species-equivariant VMMC
    ("hop_repeated_species",        (true,  true,  true )),   # species reduction with repeated multiplicities [1,1,2,2]
    ("broken_species_halfbeta",     (true,  false, true )),   # species-dependent accept -> declined, DB FAIL caught
    ("swap_literal_species",        (true,  true,  false)),   # raw-label write -> covariance guard declines
]

function _run_pipeline(n, types, algo, energy)
    states = enumerate_states(types, n)
    bfs = build_transitions(algo, energy, states, n, 30)
    erg = check_ergodicity(bfs, seed_state(types, n))
    m = build_dbmodel(bfs, energy)
    pass_r, _, ch_r = run_db_check(m; use_symmetry=true)            # graph-symmetry reduced
    pass_f, _, ch_f = run_db_check(m; use_symmetry=false)           # every pair (baseline)
    cnt = length(states) == theoretical_count(types, n)
    (tau=bfs.tau_free, db=pass_r, erg=erg.ergodic, count=cnt,
     db_full=pass_f, ch_eq=(ch_r == ch_f), sym=m.sym_names,
     npairs=length(m.pairs), nreps=length(m.check_pairs),
     species_free=bfs.species_free, nbfs=bfs.nbfs)
end

# The species (combined-orbit) BFS reduction must produce EXACTLY the same
# transition graph as a direct build with the reduction OFF. The DB-pair `==full`
# test cannot see this (it reuses one graph), so we compare the two graphs' actual
# transition probabilities at random coupling points — an independent check of the
# species-relabel + atom-permute expansion. Returns (graphs_equal, db_equal,
# species_free_on, nbfs_on, nbfs_off).
function _species_graph_consistency(n, types, algo, energy)
    states = enumerate_states(types, n)
    bon  = build_transitions(algo, energy, states, n, 30; species=true)
    boff = build_transitions(algo, energy, states, n, 30; species=false)
    atoms = Set{Atom}()
    for b in (bon, boff), lf in b.uweights, f in lf.factors; _collect_atoms_th!(atoms, f.thr); end
    tmat(b, J) = (T = Dict{Tuple{Int,Int},Float64}();
                  for (s,d,wi) in b.trans; T[(s,d)] = get(T,(s,d),0.0) + eval_leaf(b.uweights[wi], J, 1.0); end; T)
    geq = true
    Random.seed!(12345)
    for _ in 1:4
        J = Dict{Atom,Float64}(a => rand()*2 - 1 for a in atoms)
        T1 = tmat(bon, J); T2 = tmat(boff, J)
        (keys(T1) == keys(T2)) || (geq = false)
        for k in keys(T1); abs(T1[k] - get(T2, k, 0.0)) < 1e-7 || (geq = false); end
    end
    db_on,  _, _ = run_db_check(build_dbmodel(bon, energy))
    db_off, _, _ = run_db_check(build_dbmodel(boff, energy))
    (geq, db_on == db_off, bon.species_free, bon.nbfs, boff.nbfs)
end
function run_example(path)
    Base.include(Main, path)                         # (re)defines NGRID/energy/algorithm
    # invokelatest: the bindings just (re)defined by include() are newer than this
    # function's world age, so both reading them and calling them must happen in
    # the latest world — do it all inside the invokelatest closure.
    Base.invokelatest() do
        _run_pipeline(Main.NGRID, Main.PARTICLE_TYPES, Main.algorithm, Main.energy)
    end
end

@testset "EXAMPLES end-to-end: $(name)" for (name, (etau, edb, eerg)) in EXPECT
    r = run_example(joinpath(@__DIR__, "examples", name * ".jl"))
    @test r.count                # state count matches the combinatorial formula
    @test r.tau == etau
    @test r.db  == edb
    @test r.erg == eerg
    # SOUNDNESS OF THE SYMMETRY REDUCTION: the graph-symmetry-reduced DB verdict
    # MUST equal the all-pairs baseline (no false PASS, no false FAIL), with the
    # same chamber count, and it must actually reduce when a symmetry holds.
    @test r.db == r.db_full
    @test r.ch_eq
    @test r.nreps <= r.npairs
end

# Point-group symmetry and detailed balance are logically INDEPENDENT:
#  - an anisotropic defect may break BOTH point-group symmetry AND DB;
#  - an isotropic defect breaks DB but keeps full p4m symmetry;
#  - a symmetric correct move has full symmetry and satisfies DB;
#  - an anisotropic CORRECT move may have only a subgroup (D2, not D4) yet still DB-PASS.
# In all cases the symmetry reduction is speed-only: it can only reduce pair-checks,
# NEVER alter the DB verdict. "Symmetry verified" does not mean "DB holds"; "symmetry
# not verified" does not mean "DB fails".
@testset "Point-group ⟂ DB: reduction is speed-only, never a verdict" begin
    # Anisotropic broken: breaks D4 AND DB (directional bias breaks both).
    bias = run_example(joinpath(@__DIR__, "examples", "broken_biased_direction.jl"))
    @test bias.db == false                                   # DB violated
    @test !("rotate90" in bias.sym)                          # anisotropy: D4 not verified
    @test bias.db == bias.db_full                            # DB still caught despite partial sym

    # Isotropic broken: full D4 symmetry verified, yet DB fails.
    pool = run_example(joinpath(@__DIR__, "examples", "broken_variable_pool.jl"))
    @test pool.db == false                                   # DB violated
    @test ("rotate90" in pool.sym) && ("rotate180" in pool.sym)  # isotropic: full D4 verified
    @test pool.db == pool.db_full                            # D4-PASS does NOT hide the DB failure

    # Symmetric correct: full D4 symmetry, DB holds.
    good = run_example(joinpath(@__DIR__, "examples", "single_metropolis.jl"))
    @test good.db && ("rotate90" in good.sym) && ("reflect" in good.sym)
    @test good.nreps < good.npairs                           # full p4m reduces work

    # CRITICAL: D2 (not D4) correct algorithm -- key independence example.
    # Horizontal-only Metropolis has D2 (rotate180, reflect_h, reflect_v) but NOT D4
    # (rotate90 maps column-moves to row-moves, outside the proposal set). Yet DB holds.
    # This proves D4-FAIL can never be used to conclude DB-FAIL.
    hm = run_example(joinpath(@__DIR__, "examples", "horizontal_metropolis.jl"))
    @test hm.db == true                                      # DB PASSES despite not being D4
    @test !("rotate90" in hm.sym)                            # rotate90 correctly not verified
    @test !("reflect" in hm.sym)                             # diagonal reflect also not verified
    @test ("rotate180" in hm.sym) && ("reflect_h" in hm.sym) && ("reflect_v" in hm.sym)
    @test hm.erg == false                                    # ergodic FAIL by design (rows fixed)
    @test hm.nreps < hm.npairs                               # D2 still gives real pair reduction
    @test hm.db == hm.db_full                                # reduced == full check
end

# ----------------------------------------------------------------------------
# Tier 1 / Tier 2 weight classes: the new (1+exp) denominators, polynomial-
# coefficient weights, and th_max. Each is checked in BOTH directions — a correct
# instance must PASS, a deliberately broken one must be CAUGHT — so the extensions
# cannot introduce a false PASS.
@testset "Tier1/Tier2 weight classes: PASS verified, FAIL caught" begin
    n = 3; energy = mk_energy(n, 2); states = enumerate_states([1,2,3], n)
    runDB(step) = run_db_check(build_dbmodel(build_transitions(step, energy, states, n, 30), energy))[1]
    negf(lf) = LinForm(a => -c for (a, c) in lf)
    base(np_fn) = (rng, st::PState) -> begin
        i = rand_choice_index!(rng, length(st)); p = st[i]
        (dr, dc) = rand_choice!(rng, DISPS8); np = Particle(p.r + dr, p.c + dc, p.t)
        rest = st[setdiff(1:length(st), i)]
        for q in rest; same_site(q, np, n) && return st; end
        np_fn(rng, st, rest, np)
    end

    # (1+exp) denominator — Barker acceptance.  Correct => PASS.
    barker = base((rng, st, rest, np) -> begin
        ns = vcat(rest, [np]); dE = linsub(energy(ns), energy(st))
        accept!(rng, th_div(th_const(1), th_add(th_const(1), th_boltz(negf(dE))))) ? ns : st
    end)
    @test runDB(barker)
    # Broken Barker (half exponent) => must be caught.
    barker_half = base((rng, st, rest, np) -> begin
        ns = vcat(rest, [np]); dE = linsub(energy(ns), energy(st))
        h = LinForm(a => c*(1//2) for (a, c) in dE)
        accept!(rng, th_div(th_const(1), th_add(th_const(1), th_boltz(negf(h))))) ? ns : st
    end)
    @test !runDB(barker_half)

    # Polynomial-coefficient weight — a rate factor `a`.  Correct => PASS.
    rate() = th_linear(LinForm(Xparam(:a) => TauNum(1)))
    poly_ok = base((rng, st, rest, np) -> begin
        accept!(rng, rate()) || return st
        ns = vcat(rest, [np]); dE = linsub(energy(ns), energy(st))
        metropolis!(rng, dE) ? ns : st
    end)
    @test runDB(poly_ok)
    # Broken polynomial: duplicate one direction so the rate-weighted forward and
    # reverse differ.  The residual is a NON-zero polynomial in `a` => must be caught.
    biased = [(0,1),(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    poly_bad = (rng, st::PState) -> begin
        i = rand_choice_index!(rng, length(st)); p = st[i]
        (dr, dc) = rand_choice!(rng, biased); np = Particle(p.r + dr, p.c + dc, p.t)
        rest = st[setdiff(1:length(st), i)]
        for q in rest; same_site(q, np, n) && return st; end
        accept!(rng, rate()) || return st
        ns = vcat(rest, [np]); dE = linsub(energy(ns), energy(st))
        metropolis!(rng, dE) ? ns : st
    end
    @test !runDB(poly_bad)
end

@testset "th_max selects the complementary branch to th_min" begin
    _init_threadlocal!()
    A1 = Atom(true,1,1,1,:_); A2 = Atom(true,2,2,1,:_)
    a = th_boltz(LinForm(A1 => TauNum(1))); b = th_boltz(LinForm(A2 => TauNum(1)))
    mn = th_min(a, b); mx = th_max(a, b)
    aidx = Dict(A1 => 1, A2 => 2); nA = 2
    @test min_condition(a, b, aidx, nA) !== nothing            # switch is a genuine hyperplane
    ctx = (cidx = Dict{Tuple{Vector{Q},Bool},Int}(),
           thmin = Dict(objectid(mn) => 1, objectid(mx) => 1))  # sigma[1]==1 iff a<b
    va = eval_static(a, aidx, nA); vb = eval_static(b, aidx, nA)
    for σ in ([0], [1])
        vmin = eval_val(mn, σ, ctx, aidx, nA)
        vmax = eval_val(mx, σ, ctx, aidx, nA)
        @test vmin.num != vmax.num                             # they pick different operands
        @test val_add(vmin, vmax, nA).num == val_add(va, vb, nA).num   # min + max == a + b
    end
end

# Species-permutation symmetry (Step-4 graph-verified, atom-relabeling action).
@testset "species (type) permutation symmetry" begin
    n = 3; energy = mk_energy(n, 2)
    model(types, step) = build_dbmodel(build_transitions(step, energy, enumerate_states(types, n), n, 30), energy)
    has_type(m) = any(s -> startswith(s, "type"), m.sym_names)
    eqfull(m) = run_db_check(m)[1] == run_db_check(m; use_symmetry=false)[1]

    # [1,2,3]: type-equivariant single-particle Metropolis -> full S3 (both transpositions),
    # giving a strict pair reduction beyond p4m, and the verdict is unchanged.
    m123 = model([1,2,3], mk_metropolis(n, 2, energy))
    @test ("type(1<->2)" in m123.sym_names) && ("type(2<->3)" in m123.sym_names)
    @test length(m123.check_pairs) < length(m123.pairs)
    @test eqfull(m123)

    # [1,1,2]: the multiplicities differ, so NO species permutation preserves the
    # state multiset -> no species symmetry is even a candidate.
    m112 = model([1,1,2], mk_metropolis(n, 2, energy))
    @test !has_type(m112)
    @test eqfull(m112)

    # type-DEPENDENT algorithm (only species 1 ever moves). This privileges species
    # 1 but treats species 2 and 3 identically, so its actual species symmetry is
    # exactly the subgroup that FIXES species 1, i.e. swapping the spectators 2<->3.
    # The checker discovers precisely that: type(2<->3) verifies, type(1<->2) does
    # not -- and the DB verdict is unchanged (reduced == full) either way.
    only1 = (rng, st::PState) -> begin
        idxs = Int[i for i in 1:length(st) if st[i].t == 1]      # absolute-type branch
        isempty(idxs) && return st
        i = rand_choice!(rng, idxs); p = st[i]
        (dr, dc) = rand_choice!(rng, DISPS8); np = Particle(p.r + dr, p.c + dc, p.t)
        rest = st[setdiff(1:length(st), i)]
        for q in rest; same_site(q, np, n) && return st; end
        ns = vcat(rest, [np]); dE = linsub(energy(ns), energy(st))
        metropolis!(rng, dE) ? ns : st
    end
    m_only1 = model([1,2,3], only1)
    @test ("type(2<->3)" in m_only1.sym_names)                   # spectator swap IS a symmetry
    @test !("type(1<->2)" in m_only1.sym_names)                  # privileged species is not swappable
    @test eqfull(m_only1)                                        # verdict still correct
end

# Species (combined-orbit) BFS reduction — the deep soundness check: the reduced
# graph must equal a direct build, and the certificate must engage/decline exactly
# when the algorithm is/ isn't species-equivariant at the leaf level.
@testset "species BFS reduction: graph == direct build, and engages correctly" begin
    run(path) = (Base.include(Main, path);
        Base.invokelatest(() -> _species_graph_consistency(Main.NGRID, Main.PARTICLE_TYPES,
                                                           Main.algorithm, Main.energy)))
    # (name, expected species_free) — covers: distinct-S3, S3 swap, big-tree S3,
    # S2, repeated-multiplicity S2, S2 broken-but-equivariant; and the three DECLINE
    # paths (type tie-break, absolute-type branch, raw-label covariance).
    cases = [("single_metropolis", true), ("kawasaki", true), ("vmmc_2d_shuffle", true),
             ("metropolis_4x4", true), ("hop_repeated_species", true),
             ("broken_variable_pool", true),
             ("vmmc_2d", false), ("broken_species_halfbeta", false),
             ("swap_literal_species", false)]
    @testset "$(name)" for (name, exp_species) in cases
        geq, dbeq, sfree, nbfs_on, nbfs_off = run(joinpath(@__DIR__, "examples", name * ".jl"))
        @test geq                       # reduced graph == direct graph (random coupling points)
        @test dbeq                      # same DB verdict either way
        @test sfree == exp_species      # certificate engaged iff species-equivariant
        if sfree
            @test nbfs_on < nbfs_off    # the reduction actually BFS'd fewer states
        end
    end

    # ROBUSTNESS: a label operation that is not overloaded on TypeTag (here `÷`)
    # makes the tagged probe MethodError. That must NOT crash the run — it must be
    # caught, species declined, and the verdict computed correctly via the fallback.
    n = 3; energy = mk_energy(n, 2); states = enumerate_states([1,2,3], n)
    weird = (rng, st::PState) -> begin
        pidx = rand_choice_index!(rng, length(st)); p = st[pidx]
        _ = p.t ÷ 2                                   # un-overloaded op on a label
        (dr, dc) = rand_choice!(rng, DISPS8); np = Particle(p.r+dr, p.c+dc, p.t)
        rest = st[setdiff(1:length(st), pidx)]
        for q in rest; same_site(q, np, n) && return st; end
        ns = vcat(rest, [np]); dE = linsub(energy(ns), energy(st))
        metropolis!(rng, dE) ? ns : st
    end
    local bfs
    @test (bfs = build_transitions(weird, energy, states, n, 30)) isa BFSResult   # no crash
    @test bfs.species_free == false                                               # declined
    p_on, _, _  = run_db_check(build_dbmodel(bfs, energy))
    p_off, _, _ = run_db_check(build_dbmodel(build_transitions(weird, energy, states, n, 30; species=false), energy))
    @test p_on == p_off                                                           # correct via fallback
end

end
