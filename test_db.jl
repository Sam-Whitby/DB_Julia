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
    wRev  = val_oneminus(val_boltz(Lrev), nA)
    wFwd  = val_oneminus(val_boltz(Lfwd), nA)
    ratio = Val(wRev.num, [Lfwd])
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
]

function _run_pipeline(n, types, algo, energy)
    states = enumerate_states(types, n)
    bfs = build_transitions(algo, energy, states, n, 30)
    erg = check_ergodicity(bfs, seed_state(types, n))
    pass, _, _ = run_db_check(build_dbmodel(bfs, energy))
    cnt = length(states) == theoretical_count(types, n)
    (tau=bfs.tau_free, db=pass, erg=erg.ergodic, count=cnt)
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
end

end
