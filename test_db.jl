# ============================================================================
# test_db.jl  —  Regression suite for DB_Julia
# ============================================================================
# Proves the individual pieces (TauNum, bit-exact selection weights, the exact
# rational-function engine, tau detection) and runs fast end-to-end checks with
# both PASS and FAIL outcomes, so a false positive OR a false negative would
# fail the suite.
#
# Run:  julia --project=. test_db.jl
# ============================================================================

include(joinpath(@__DIR__, "dbc.jl"))
using Test, Random

@testset "DB_Julia" begin

@testset "TauNum: linear tracking + nonlinear taint" begin
    a = tau_r_aug(3); b = tau_r_aug(1)
    @test is_tau_free(a - b) && tau0(a - b) == 2
    @test !is_tau_free(a^2) && tau0(a^2) == 9
    @test !is_tau_free(a + 5) && tau0(a + 5) == 8
    @test is_tau_free((a - b) * tau_const(4))
    @test !is_tau_free(tau_r_aug(1) * tau_c_aug(1))
end

@testset "nbits matches IntegerLength[n-1,2]" begin
    @test [nbits(n) for n in 1:9] == [0,1,2,2,3,3,3,3,4]
end

@testset "rejection-sampling weights are exact (1/n), n leaves" begin
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

@testset "Val engine: VMMC ratio cancels exactly" begin
    nA = 2; Lfwd = Q[1,0]; Lrev = Q[0,1]
    wRev  = val_oneminus(val_boltz(Lrev), nA)              # 1 - exp(-b Lrev)
    wFwd  = val_oneminus(val_boltz(Lfwd), nA)
    ratio = Val(wRev.num, [Lfwd])                          # (1-exp(-bLrev))/(1-exp(-bLfwd))
    link  = val_mul(wFwd, ratio)                           # the (1-exp(-bLfwd)) cancels
    # link == wRev as a rational function: cross-multiply.
    @test bs_mul(link.num, expand_binoms(wRev.den, nA)) == bs_mul(wRev.num, expand_binoms(link.den, nA))
    D = ms_unionmax(link.den, wRev.den)
    res = bs_sub(bs_mul(link.num, expand_binoms(ms_diff(D,link.den),nA)),
                 bs_mul(wRev.num, expand_binoms(ms_diff(D,wRev.den),nA)))
    @test isempty(res)
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

@testset "leaf weights sum to 1 at random coupling points" begin
    n=3; energy = mk_energy(n,2); step = mk_metropolis(n,2,energy)
    _TAU[] = false
    leaves = build_state_leaves(step, augmented_pstate(sort([(1,1,1),(2,2,2),(3,3,3)])), n, 22)
    @test _TAU[] == false
    atoms = Set{Atom}(); for l in leaves, f in l.factors; _collect_atoms_th!(atoms, f.thr); end
    Random.seed!(1)
    for _ in 1:5
        J = Dict{Atom,Float64}(a => rand()*2-1 for a in atoms)
        @test abs(sum(eval_leaf(l, J, 1.0) for l in leaves) - 1.0) < 1e-9
    end
end

@testset "tau detection: no false positive, no false negative" begin
    n=2
    energy_qf = function(state::PState)
        lf = mk_energy(n,2)(state)
        fld = TauNum(0); for p in state; fld = fld + p.r^2; end
        addcoef!(lf, Xparam(:fieldH), fld); lf
    end
    seed = augmented_pstate(sort([(1,1,1),(1,2,2)]))
    _TAU[]=false; build_state_leaves(mk_metropolis(n,2,energy_qf), seed, n, 22); @test _TAU[] == true
    _TAU[]=false; build_state_leaves(mk_metropolis(n,2,mk_energy(n,2)), seed, n, 22); @test _TAU[] == false
end

@testset "end-to-end DB PASS (correct metropolis, nGrid=2)" begin
    n=2; energy = mk_energy(n,2); states = enumerate_states([1,2], n)
    bfs = build_transitions(mk_metropolis(n,2,energy), energy, states, n, 22)
    pass, _, _ = run_db_check(build_dbmodel(bfs, energy))
    @test bfs.tau_free && pass
end

@testset "end-to-end DB FAIL (variable-pool 4-hop)" begin
    n=3; disps = [(-1,0),(1,0),(0,-1),(0,1)]
    energy0 = (state::PState) -> LinForm()
    bvp = function(rng, state::PState)
        pidx = rand_choice_index!(rng, length(state)); p = state[pidx]
        rest = state[setdiff(1:length(state), pidx)]
        valid = [(d1,d2) for (d1,d2) in disps
                 if !any(same_site(q, Particle(p.r+d1,p.c+d2,p.t), n) for q in rest)]
        isempty(valid) && return state
        (d1,d2) = rand_choice!(rng, valid); vcat(rest, [Particle(p.r+d1, p.c+d2, p.t)])
    end
    bfs = build_transitions(bvp, energy0, enumerate_states([1,2], n), n, 22)
    pass, _, _ = run_db_check(build_dbmodel(bfs, energy0))
    @test bfs.tau_free && !pass
end

@testset "exact rationals: half-beta exponent is FAIL" begin
    n=3; energy = mk_energy(n,2)
    half = function(rng, state::PState)
        pidx = rand_choice_index!(rng, length(state)); p = state[pidx]
        (dr,dc) = rand_choice!(rng, DISPS8); newp = Particle(p.r+dr, p.c+dc, p.t)
        rest = state[setdiff(1:length(state), pidx)]
        for q in rest; same_site(q, newp, n) && return state; end
        ns = vcat(rest, [newp]); dE = linsub(energy(ns), energy(state))
        metropolis!(rng, dE; exponent = LinForm(a => c*(1//2) for (a,c) in dE)) ? ns : state
    end
    bfs = build_transitions(half, energy, enumerate_states([1,2,3], n), n, 22)
    pass, _, _ = run_db_check(build_dbmodel(bfs, energy))
    @test !pass
end

end
