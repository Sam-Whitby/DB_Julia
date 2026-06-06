# ============================================================================
# cluster_metropolis.jl  —  a cluster move that DOES satisfy detailed balance,
# because the cluster's interaction with its environment is accounted for in a
# FINAL Metropolis acceptance (the original Whitelam–Geissler idea).
# ============================================================================
# This is the correct counterpart to the deliberately-broken vmmc_early_stop.jl.
# There, cluster growth was terminated early with NO compensation, so the move did
# not sample the Boltzmann distribution (DB FAIL, balance FAIL). Here the cluster is
# likewise small (it stops after one shell of recruitment — "early stopping"), but a
# final acceptance step makes it exact.
#
# The move (a depth-1 virtual-move cluster):
#   1. Pick a seed particle and a displacement `dir` (from the declared MOVES).
#   2. RECRUIT each occupied in-range neighbour of the seed into the cluster, each
#      independently with probability  q = 1 - exp(-beta*mu),  where `mu >= 0` is a
#      "recruitment" coupling. (Drawn via accept! on the symbolic threshold, so the
#      checker tracks it exactly; written through `unordered`, so the recruitment is
#      order-independent and the species/point-group reductions still apply.)
#   3. Translate the whole cluster by `dir`; reject on hard-core overlap.
#   4. ACCEPT with a FINAL Metropolis step  min(1, exp(-beta * dE_eff)),  where
#         dE_eff = [E(new) - E(old)]            # cluster-vs-environment energy change
#                + mu * (n_t - n_s)             # recruitment proposal-ratio correction
#      and n_s / n_t are the seed's in-range occupied-neighbour counts BEFORE / AFTER
#      the move. The `mu*(n_t - n_s)` term is exactly the ratio of forward/reverse
#      recruitment proposals ((1-q)^(n_t - n_s) = exp(-beta*mu*(n_t-n_s))); together
#      with the Boltzmann factor this is the Metropolis–Hastings acceptance, so
#      detailed balance holds for ALL couplings (and `mu >= 0` makes q a probability).
#
# Without the `mu*(n_t - n_s)` correction the proposal asymmetry (the seed has a
# different number of neighbours before and after the move) would break DB — that
# missing piece is precisely what a "final Metropolis acceptance against the
# environment" supplies, and what vmmc_early_stop.jl omits.
#
# Expected result:  tau PASS,  detailed balance PASS,  global balance PASS,
#                   ergodicity PASS  (species S3 + point group D4).
# ============================================================================

const NGRID          = 3
const MAXD2          = 2
const PARTICLE_TYPES = [1, 2, 3]
const MOVES          = [(dx, dy) for dx in -1:1 for dy in -1:1 if (dx, dy) != (0, 0)]

function energy(state::PState)::LinForm
    lf = LinForm()
    for i in 1:length(state), j in (i+1):length(state)
        d2 = pbc_d2(state[i], state[j], NGRID)
        (0 < d2 <= MAXD2) || continue
        addcoef!(lf, Jc(state[i].t, state[j].t, d2), TauNum(1))
    end
    lf
end

inrange(a::Particle, b::Particle) = 0 < pbc_d2(a, b, NGRID) <= MAXD2

function algorithm(rng, state::PState)::PState
    isempty(state) && return state
    seedidx = rand_choice_index!(rng, length(state))
    dir     = rand_move!(rng)
    seed    = state[seedidx]

    # candidates = the seed's in-range occupied neighbours (n_s of them)
    cands = Int[qi for qi in 1:length(state) if qi != seedidx && inrange(state[qi], seed)]
    n_s   = length(cands)

    # recruit each with probability q = 1 - exp(-beta*mu)
    mu   = LinForm(); addcoef!(mu, Xparam(:mu), TauNum(1))
    qthr = th_sub(th_const(1), th_boltz(mu))
    cluster = [seedidx]
    for qi in unordered(rng, cands)
        accept!(rng, qthr) && push!(cluster, qi)
    end

    # translate the cluster; hard-core rejection against the environment
    clset      = Set(cluster)
    noncluster = eltype(state)[state[i] for i in 1:length(state) if !(i in clset)]
    moved      = eltype(state)[move(state[ci], dir) for ci in cluster]
    for mp in moved
        for q in noncluster
            same_site(q, mp, NGRID) && return state
        end
    end
    newstate = vcat(noncluster, moved)

    # final Metropolis: energy change + recruitment proposal-ratio correction
    seedmoved = move(seed, dir)
    n_t = count(q -> inrange(q, seedmoved), newstate)        # seed's neighbours after the move
    dE  = linsub(energy(newstate), energy(state))
    addcoef!(dE, Xparam(:mu), TauNum(n_t - n_s))             # + mu*(n_t - n_s)
    metropolis!(rng, dE) ? newstate : state
end
