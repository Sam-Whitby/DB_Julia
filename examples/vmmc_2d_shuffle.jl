# ============================================================================
# vmmc_2d_shuffle.jl  —  VMMC with a RANDOM candidate order (no canonical sort)
# ============================================================================
# Identical to vmmc_2d.jl except that, instead of processing each cluster
# particle's candidate neighbours in a fixed canonical order (a sort whose final
# tie-break is the species label `state[qi].t`), this version visits them in a
# UNIFORMLY RANDOM order drawn from the rng (a Fisher-Yates shuffle).
#
# Why: randomising the consideration order is what most real Monte-Carlo codes do,
# and it removes the type-dependent tie-break. Consequences:
#   * the decision tree is LARGER (the permutation consumes extra random bits), so
#     the tau-BFS explores more paths;
#   * the algorithm becomes SPECIES-EQUIVARIANT (nothing branches on the absolute
#     type label any more), so — unlike vmmc_2d — the species-permutation symmetry
#     is now verifiable (S3 here).
# The candidate SET is unchanged and each per-candidate link/frustration decision
# is independent of order, so the move's transition probabilities are unchanged and
# detailed balance still holds.
#
# Expected result:  tau PASS,  DB PASS,  ERGODICITY PASS  (and species S3 verified)
# ============================================================================

const NGRID          = 3
const MAXD2          = 2
const PARTICLE_TYPES = [1, 2, 3]

const VMMC_DISPS = [(dx, dy) for dx in -1:1 for dy in -1:1 if (dx, dy) != (0, 0)]

function energy(state::PState)::LinForm
    lf = LinForm()
    for i in 1:length(state), j in (i+1):length(state)
        d2 = pbc_d2(state[i], state[j], NGRID)
        (0 < d2 <= MAXD2) || continue
        addcoef!(lf, Jc(state[i].t, state[j].t, d2), TauNum(1))
    end
    lf
end

function pairE(ti, tj, pi::Particle, pj::Particle, n::Int)::LinForm
    lf = LinForm()
    d2 = pbc_d2(pi, pj, n)
    (0 < d2 <= MAXD2) && addcoef!(lf, Jc(ti, tj, d2), TauNum(1))
    lf
end

shift(p::Particle, d) = Particle(p.r + d[1], p.c + d[2], p.t)

wfwd_threshold(eInit::LinForm, eFwd::LinForm) =
    th_piece([([c_lt(eInit, eFwd)],
               th_sub(th_const(1), th_boltz(linsub(eFwd, eInit))))], th_const(0))

function pw_threshold(eInit::LinForm, eFwd::LinForm, eRev::LinForm)
    ratio = th_div(th_sub(th_const(1), th_boltz(linsub(eRev, eInit))),
                   th_sub(th_const(1), th_boltz(linsub(eFwd, eInit))))
    th_piece([([c_lt(eInit, eFwd), c_lt(eInit, eRev)], th_min(ratio, th_const(1))),
              ([c_lt(eInit, eFwd)], th_const(0))], th_const(0))
end

# Fisher-Yates shuffle using the rng (exact uniform over orderings; each swap draws
# a uniform integer with exact rejection-sampling weight). Species-blind.
function shuffle_rng!(rng, v::Vector{Int})
    for i in length(v):-1:2
        j = rand_integer!(rng, 1, i)
        v[i], v[j] = v[j], v[i]
    end
    v
end

function build_cluster(rng, state::PState, n::Int, seedidx::Int, dir)
    cluster = [seedidx]; incluster = Set(cluster); queue = [seedidx]
    while !isempty(queue)
        pidx = popfirst!(queue); p = state[pidx]
        pPost = shift(p, dir); pRev = shift(p, (-dir[1], -dir[2]))
        cands = Int[]
        for qi in 1:length(state)
            (qi in incluster) && continue
            q = state[qi]
            dP = pbc_d2(q, p, n); dPost = pbc_d2(q, pPost, n); dRev = pbc_d2(q, pRev, n)
            ((0 < dP <= MAXD2) || (0 < dPost <= MAXD2) || (0 < dRev <= MAXD2)) && push!(cands, qi)
        end
        shuffle_rng!(rng, cands)                 # RANDOM order (no type-dependent sort)
        for qi in cands
            q = state[qi]
            eInit = pairE(p.t, q.t, p,     q, n)
            eFwd  = pairE(p.t, q.t, pPost, q, n)
            eRev  = pairE(p.t, q.t, pRev,  q, n)
            if accept!(rng, wfwd_threshold(eInit, eFwd))
                if !accept!(rng, pw_threshold(eInit, eFwd, eRev))
                    return :frustrated
                end
                push!(cluster, qi); push!(incluster, qi); push!(queue, qi)
            end
        end
    end
    cluster
end

function algorithm(rng, state::PState)::PState
    isempty(state) && return state
    seedidx = rand_choice_index!(rng, length(state))
    dir     = rand_choice!(rng, VMMC_DISPS)

    cl = build_cluster(rng, state, NGRID, seedidx, dir)
    cl === :frustrated && return state

    clset = Set(cl)
    noncluster = eltype(state)[state[i] for i in 1:length(state) if !(i in clset)]
    for ci in cl
        dest = shift(state[ci], dir)
        for q in noncluster
            same_site(q, dest, NGRID) && return state
        end
    end
    vcat(noncluster, eltype(state)[shift(state[ci], dir) for ci in cl])
end
