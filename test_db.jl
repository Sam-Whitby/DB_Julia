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

_pre_dbc = Set(names(Main; all=true))
include(joinpath(@__DIR__, "dbc.jl"))
# Names dbc.jl added to Main — injected into each example's fresh module below.
const _DBC_NAMES = setdiff(names(Main; all=true), _pre_dbc)
using Test, Random

# Load an example into a FRESH module so its `const NGRID` / `PARTICLE_TYPES` /
# `MOVES` never collide or go stale across includes (redefining a `const` global in
# Main is unreliable across many includes — it silently kept a previous example's
# value, e.g. running quadratic_field's 2-particle/2x2 case on a stale 3-particle/3x3
# state). The fresh module gets Base (default) plus every dbc binding by value, so
# the example's `pbc_d2`, `Jc`, `move`, `Particle`, … all resolve.
function load_example(path)
    m = Module(gensym("Ex"))
    for nm in _DBC_NAMES
        s = string(nm)
        (startswith(s, "#") || startswith(s, "@")) && continue
        isdefined(Main, nm) || continue
        try; Core.eval(m, :($nm = $(getfield(Main, nm)))); catch; end
    end
    Base.include(m, path)
    m
end

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
    ("vmmc_2d_unordered",           (true,  true,  true )),   # `unordered` primitive (OIP) -> species + D4, no shuffle blow-up
    ("hop_repeated_species",        (true,  true,  true )),   # species reduction with repeated multiplicities [1,1,2,2]
    ("broken_species_halfbeta",     (true,  false, true )),   # species-dependent accept -> declined, DB FAIL caught
    ("swap_literal_species",        (true,  true,  false)),   # raw-label write -> covariance guard declines
    ("directed_sweep",              (true,  false, false)),   # non-reversible: DB FAIL (balance PASS, see balance testset)
    ("vmmc_early_stop",             (true,  false, true )),   # order-INDEP early stop: OIP accepts, DB FAIL (balance FAIL too)
    ("cluster_metropolis",          (true,  true,  true )),   # early stop FIXED by a final Metropolis vs environment -> DB PASS
]

function _run_pipeline(n, types, algo, energy; moves=nothing)
    states = enumerate_states(types, n)
    bfs = build_transitions(algo, energy, states, n, 30; moves=moves)
    erg = check_ergodicity(bfs, seed_state(types, n))
    m = build_dbmodel(bfs, energy)
    pass_r, _, ch_r = run_db_check(m; use_symmetry=true)            # graph-symmetry reduced
    pass_f, _, ch_f = run_db_check(m; use_symmetry=false)           # every pair (baseline)
    cnt = length(states) == theoretical_count(types, n)
    (tau=bfs.tau_free, db=pass_r, erg=erg.ergodic, count=cnt,
     db_full=pass_f, ch_eq=(ch_r == ch_f), sym=m.sym_names,
     npairs=length(m.pairs), nreps=length(m.check_pairs),
     species_free=bfs.species_free, nbfs=bfs.nbfs, pg_idx=bfs.pg_idx)
end

# The species (combined-orbit) BFS reduction must produce EXACTLY the same
# transition graph as a direct build with the reduction OFF. The DB-pair `==full`
# test cannot see this (it reuses one graph), so we compare the two graphs' actual
# transition probabilities at random coupling points — an independent check of the
# species-relabel + atom-permute expansion. Returns (graphs_equal, db_equal,
# species_free_on, nbfs_on, nbfs_off).
function _species_graph_consistency(n, types, algo, energy; moves=nothing)
    states = enumerate_states(types, n)
    # Isolate the SPECIES reduction (point group off on both sides) so this checks the
    # species-relabel + atom-permute expansion specifically. `moves` is still passed
    # so rand_move!/move work for contract examples; use_pointgroup=false keeps H={id}.
    bon  = build_transitions(algo, energy, states, n, 30; species=true,  moves=moves, use_pointgroup=false)
    boff = build_transitions(algo, energy, states, n, 30; species=false, moves=moves, use_pointgroup=false)
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

# Point-group (translation × species × p4m) BFS reduction — soundness check: the
# fully-reduced graph must equal an INDEPENDENT baseline at random coupling points.
# The baseline is the translation-only build (species & point-group OFF), which BFSes
# every translation rep separately and derives by PURE translation — a different
# derivation path from translate∘rotate∘relabel, and itself anchored to a direct
# all-states BFS by the species test. With `direct=true` the baseline is instead a
# full direct all-states BFS (no derivation at all) — the gold standard, used on the
# small example. Returns (graph_equal, db_pass, pg_idx, nbfs_reduced, nstates).
function _pg_graph_consistency(n, types, algo, energy; moves, direct::Bool=false)
    states = enumerate_states(types, n); ix = Dict(s => i for (i, s) in enumerate(states))
    bon = build_transitions(algo, energy, states, n, 30; species=true, moves=moves, use_pointgroup=true)
    atoms = Set{Atom}(); for lf in bon.uweights, f in lf.factors; _collect_atoms_th!(atoms, f.thr); end
    tmat_on(J) = (T = Dict{Tuple{Int,Int},Float64}();
                  for (s,d,wi) in bon.trans; T[(s,d)] = get(T,(s,d),0.0) + eval_leaf(bon.uweights[wi], J, 1.0); end; T)
    if direct                                   # gold standard: BFS every state, no derivation
        _MOVESET[] = moves === nothing ? Tuple{Int,Int}[] : moves; _ROT_PROBE[] = false; _init_threadlocal!()
        base = Vector{Vector{Leaf}}(undef, length(states))
        for (i, s) in enumerate(states); base[i] = build_state_leaves(algo, augmented_pstate(s), n, 30); end
        for lvs in base, lf in lvs, f in lf.factors; _collect_atoms_th!(atoms, f.thr); end
        tmat_base(J) = (T = Dict{Tuple{Int,Int},Float64}();
                        for (i,lvs) in enumerate(base), lf in lvs; d = ix[lf.next];
                            d != i && (T[(i,d)] = get(T,(i,d),0.0) + eval_leaf(lf, J, 1.0)); end; T)
        cmp = tmat_base
    else                                        # cheap baseline: translation-only build
        btr = build_transitions(algo, energy, states, n, 30; species=false, moves=moves, use_pointgroup=false)
        for lf in btr.uweights, f in lf.factors; _collect_atoms_th!(atoms, f.thr); end
        tmat_tr(J) = (T = Dict{Tuple{Int,Int},Float64}();
                      for (s,d,wi) in btr.trans; T[(s,d)] = get(T,(s,d),0.0) + eval_leaf(btr.uweights[wi], J, 1.0); end; T)
        cmp = tmat_tr
    end
    geq = true; Random.seed!(999)
    for _ in 1:4
        J = Dict{Atom,Float64}(a => rand()*2 - 1 for a in atoms)
        T1 = tmat_on(J); T2 = cmp(J)
        for k in union(keys(T1), keys(T2)); abs(get(T1,k,0.0) - get(T2,k,0.0)) < 1e-7 || (geq = false); end
    end
    dbon, _, _ = run_db_check(build_dbmodel(bon, energy))
    (geq, dbon, bon.pg_idx, bon.nbfs, length(states))
end

function run_example(path)
    m = load_example(path)
    # invokelatest: m's functions are newer than this function's world age, so both
    # reading and calling them must happen in the latest world.
    Base.invokelatest() do
        moves = isdefined(m, :MOVES) ? Vector{Tuple{Int,Int}}(m.MOVES) : nothing
        _run_pipeline(m.NGRID, m.PARTICLE_TYPES, m.algorithm, m.energy; moves=moves)
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
    run(path) = (m = load_example(path);
        Base.invokelatest(() -> _species_graph_consistency(m.NGRID, m.PARTICLE_TYPES,
            m.algorithm, m.energy;
            moves = isdefined(m, :MOVES) ? Vector{Tuple{Int,Int}}(m.MOVES) : nothing)))
    # (name, expected species_free) — covers: distinct-S3, S3 swap, big-tree S3,
    # S2, repeated-multiplicity S2, S2 broken-but-equivariant; and the three DECLINE
    # paths (type tie-break, absolute-type branch, raw-label covariance).
    cases = [("single_metropolis", true), ("kawasaki", true), ("vmmc_2d_shuffle", true),
             ("vmmc_2d_unordered", true), ("vmmc_early_stop", true), ("cluster_metropolis", true),
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

# Point-group (p4m) BFS reduction — the gold-standard soundness + correct-subgroup
# + decline tests. The fully-reduced graph (translation × species × point-group)
# must equal a DIRECT all-states BFS, the discovered subgroup must be exactly right,
# and an algorithm that bypasses the supplied-direction contract must DECLINE the
# point group (never a silent wrong reduction) while keeping the correct verdict.
@testset "point-group BFS reduction: graph == direct build, correct subgroup, declines safely" begin
    D4 = Set(["identity","rotate90","rotate180","rotate270","reflect","reflect_h","reflect_v","reflect_ad"])
    D2 = Set(["identity","rotate180","reflect_h","reflect_v"])
    runpg(path; direct=false) = (m = load_example(path);
        Base.invokelatest(() -> _pg_graph_consistency(m.NGRID, m.PARTICLE_TYPES,
            m.algorithm, m.energy; moves = Vector{Tuple{Int,Int}}(m.MOVES), direct=direct)))
    # direct=true (gold-standard direct all-states baseline) only for the small example
    # (horizontal, 72 states) to keep the suite fast; the rest use the cheap, equally
    # sound translation-only baseline.
    cases = [("single_metropolis", D4, false), ("metropolis_4x4", D4, false),
             ("hop_8way_correct", D4, false), ("horizontal_metropolis", D2, true),
             ("vmmc_2d_shuffle", D4, false),  # species + D4
             ("vmmc_2d_unordered", D4, false),# species + D4 via the `unordered` primitive
             ("vmmc_early_stop", D4, true),   # OIP false-accept detector: reduced==DIRECT all-states
             ("cluster_metropolis", D4, false),# correct early-stop cluster (final Metropolis vs environment)
             ("vmmc_2d", D4, false),          # D4 only (sort tie-breaks species)
             ("kawasaki", D4, false)]         # empty move set -> vacuous D4, + species
    @testset "$(name)" for (name, expset, direct) in cases
        geq, db, pgidx, nbfs, nstates = runpg(joinpath(@__DIR__, "examples", name * ".jl"); direct=direct)
        @test geq                                  # full reduced graph == independent baseline
        @test Set(pt_name.(pgidx)) == expset       # exactly the right point-group subgroup
        @test nbfs < nstates                       # actually fewer BFS
    end

    # DECLINE: an algorithm that bypasses rand_move! (hardcoded direction / bare
    # position arithmetic) must trip the rotation probe -> point group DECLINED
    # (pg_idx empty), with the DB verdict unchanged from a point-group-off build.
    n = 3; energy = mk_energy(n, 2); states = enumerate_states([1,2,3], n)
    hard = (rng, st::PState) -> begin
        pidx = rand_choice_index!(rng, length(st)); p = st[pidx]
        np = move(p, (1, 0))                        # hardcoded tuple: NOT from rand_move!
        rest = st[setdiff(1:length(st), pidx)]
        for q in rest; same_site(q, np, n) && return st; end
        ns = vcat(rest, [np]); dE = linsub(energy(ns), energy(st))
        metropolis!(rng, dE) ? ns : st
    end
    bh   = build_transitions(hard, energy, states, n, 30; moves=DISPS8, use_pointgroup=true)
    bhno = build_transitions(hard, energy, states, n, 30; moves=DISPS8, use_pointgroup=false)
    @test isempty(bh.pg_idx)                                          # point group declined
    @test run_db_check(build_dbmodel(bh,   energy))[1] ==
          run_db_check(build_dbmodel(bhno, energy))[1]                # verdict unaffected

    # A non-closed direction set yields only the subgroup it IS closed under.
    @test Set(pt_name.(pointgroup_subgroup([(0,1),(0,-1)]))) == D2    # column moves -> D2
    @test Set(pt_name.(pointgroup_subgroup(DISPS8)))         == D4    # king moves   -> D4
    @test pointgroup_subgroup([(1,0)]) ⊆ collect(1:8) &&
          "rotate90" ∉ pt_name.(pointgroup_subgroup([(1,0)]))        # single dir: no rotate90
end

# Order-independent iteration primitive (`unordered`, the OIP). An order-INDEPENDENT
# body is certified (and its single-order graph equals a direct build — verified by
# the species/point-group consistency testsets above); an order-DEPENDENT body that
# misuses `unordered` must be CAUGHT by the cross-check (a hard error), never silently
# reduced. The win is that `unordered` consumes NO random bits, so it avoids the
# factorial decision-tree blow-up of an explicit shuffle while keeping the symmetry.
@testset "OIP: `unordered` certified when valid, misuse caught" begin
    n = 3; energy = mk_energy(n, 2); states = enumerate_states([1,2,3], n)

    # MISUSE: a body whose successor depends on the VISITING ORDER (it moves the seed
    # iff the first-linked spectator has an even index — and which spectator is
    # "first" depends on the order). Its transition probabilities differ between
    # orders, so the OIP cross-check must hard-error rather than reduce unsoundly.
    orderdep = (rng, st::PState) -> begin
        s = rand_choice_index!(rng, length(st)); p = st[s]
        cands = Int[i for i in 1:length(st) if i != s]
        first_linked = 0
        for qi in unordered(rng, cands)
            (first_linked == 0 && accept!(rng, th_const(1//2))) && (first_linked = qi)
        end
        rest = st[setdiff(1:length(st), s)]
        np = (first_linked != 0 && iseven(first_linked)) ? Particle(p.r, p.c + 1, p.t) : p
        for q in rest; same_site(q, np, n) && return st; end
        vcat(rest, [np])
    end
    @test_throws CantHandle build_transitions(orderdep, energy, states, n, 30)

    # MISUSE 2 (the user's early-termination scenario): a cluster builder that stops
    # MID-candidate-loop and moves the PARTIAL cluster. The partial cluster depends on
    # which candidates were visited first, so the off-diagonal transition probabilities
    # are order-dependent -> the cross-check must REJECT it. (Stopping AFTER a whole
    # particle's loop, by contrast, is order-independent and accepted — see
    # examples/vmmc_early_stop.jl.) This confirms OIP is not fooled by an early-abort
    # that reaches an off-diagonal state (not just the diagonal self-loop).
    midstop = (rng, st::PState) -> begin
        seedidx = rand_choice_index!(rng, length(st)); dir = rand_move!(rng); p0 = st[seedidx]
        cluster = [seedidx]; incluster = Set(cluster)
        pPost = move(p0, dir); cands = Int[]
        for qi in 1:length(st)
            (qi in incluster) && continue
            ((0 < pbc_d2(st[qi], p0, n) <= 2) || (0 < pbc_d2(st[qi], pPost, n) <= 2)) && push!(cands, qi)
        end
        for qi in unordered(rng, cands)
            accept!(rng, th_const(1//2)) && (push!(cluster, qi); push!(incluster, qi))
            accept!(rng, th_const(1//4)) && break          # MID-loop stop -> order-dependent partial cluster
        end
        clset = Set(cluster); noncl = eltype(st)[st[i] for i in 1:length(st) if !(i in clset)]
        for ci in cluster; dest = move(st[ci], dir); for q in noncl; same_site(q, dest, n) && return st; end; end
        vcat(noncl, eltype(st)[move(st[ci], dir) for ci in cluster])
    end
    @test_throws CantHandle build_transitions(midstop, energy, states, n, 30; moves=DISPS8)

    # VALID: the order-INDEPENDENT VMMC variant is certified end-to-end (species + D4)
    # and the reduced verdict equals the all-pairs baseline.
    r = run_example(joinpath(@__DIR__, "examples", "vmmc_2d_unordered.jl"))
    @test r.tau && r.db && r.erg                 # tau / DB / ergodicity all PASS
    @test r.species_free && !isempty(r.pg_idx)   # species AND point group both engaged
    @test r.db == r.db_full                       # symmetry reduction is verdict-neutral
    @test r.nbfs < length(states)                 # actually fewer states BFS'd
end

# Global balance (`-balance`): the COLUMN-sum condition pi*T = pi, which correct
# sampling actually requires. It is strictly WEAKER than detailed balance (a
# non-reversible chain can satisfy balance while violating DB), it is computed by the
# SAME exact rational machinery (no floats), and the state-orbit reduction must equal
# the all-targets baseline (sound; speed only).
@testset "Global balance (-balance): weaker than DB, exact, reduced == full" begin
    n = 3
    runmodes(types, algo, en; mv=nothing) = begin
        states = enumerate_states(types, n)
        m = build_dbmodel(build_transitions(algo, en, states, n, 30; moves=mv), en)
        (db       = run_db_check(m; mode=:detailed)[1],
         bal      = run_db_check(m; mode=:balance)[1],
         balfull  = run_db_check(m; mode=:balance, use_symmetry=false)[1],
         ntargets = length(m.check_targets), nstates = length(states))
    end

    # directed_sweep: the canonical non-reversible chain -> DB FAIL but BALANCE PASS.
    ds = load_example(joinpath(@__DIR__, "examples", "directed_sweep.jl"))
    r  = Base.invokelatest(() -> runmodes(ds.PARTICLE_TYPES, ds.algorithm, ds.energy))
    @test r.db == false                  # detailed balance fails (directed move)
    @test r.bal == true                  # global balance holds (cyclic permutation -> uniform stationary)
    @test r.bal == r.balfull             # state-orbit reduction == all-targets baseline
    @test r.ntargets < r.nstates         # the reduction actually reduces

    # DB ==> balance: a detailed-balance-PASS algorithm also passes balance, reduced
    # check agreeing with the full baseline.
    energy = mk_energy(n, 2)
    rr = runmodes([1,2,3], mk_metropolis(n, 2, energy), energy)
    @test rr.db == true && rr.bal == true
    @test rr.bal == rr.balfull

    # A DB-FAIL example: the column reduction is sound regardless of the verdict
    # (reduced balance == full balance), and DB-FAIL does not imply balance-PASS.
    bvp = load_example(joinpath(@__DIR__, "examples", "broken_variable_pool.jl"))
    rb  = Base.invokelatest(() -> runmodes(bvp.PARTICLE_TYPES, bvp.algorithm, bvp.energy))
    @test rb.bal == rb.balfull

    # vmmc_early_stop: an order-INDEPENDENT (OIP-accepted) move that breaks BOTH
    # conditions — naive early termination samples neither pi via DB nor via balance.
    # DB-FAIL does NOT imply balance-PASS; the checker catches the broken move both ways.
    es = load_example(joinpath(@__DIR__, "examples", "vmmc_early_stop.jl"))
    re = Base.invokelatest(() -> runmodes(es.PARTICLE_TYPES, es.algorithm, es.energy;
                                          mv = Vector{Tuple{Int,Int}}(es.MOVES)))
    @test re.db == false && re.bal == false      # both detailed balance AND global balance fail
    @test re.bal == re.balfull                    # column reduction still sound

    # cluster_metropolis: the SAME early-stopping cluster, now CORRECTED by a final
    # Metropolis acceptance against the environment (+ the recruitment proposal-ratio).
    # It satisfies detailed balance, hence also global balance.
    cm = load_example(joinpath(@__DIR__, "examples", "cluster_metropolis.jl"))
    rc = Base.invokelatest(() -> runmodes(cm.PARTICLE_TYPES, cm.algorithm, cm.energy;
                                          mv = Vector{Tuple{Int,Int}}(cm.MOVES)))
    @test rc.db == true && rc.bal == true
    @test rc.bal == rc.balfull
end

end
