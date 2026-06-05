# ============================================================================
# vmmc_2d_unordered.jl  —  VMMC with the ORDER-INDEPENDENT PRIMITIVE (`unordered`)
# ============================================================================
# Identical physics to vmmc_2d.jl / vmmc_2d_shuffle.jl, but the per-particle
# candidate loop is processed via `unordered(rng, cands)` — the checker's
# order-independent iteration primitive (the "OIP contract").
#
# Why this is the best of both worlds:
#   * vmmc_2d.jl    sorts candidates with a tie-break on the species label
#                   `state[qi].t`, so it DECLINES the species symmetry.
#   * vmmc_2d_shuffle.jl  removes the tie-break by drawing a uniformly random order
#                   with the rng (Fisher-Yates), which RESTORES species-equivariance
#                   but BLOWS UP the decision tree by |cands|! (the shuffle consumes
#                   extra random bits at every cluster step), so the tau-BFS explores
#                   millions of redundant order-permuted paths.
#   * THIS file     uses `unordered`, which yields a canonical order that reads no
#                   absolute label (so species/point-group symmetry is preserved)
#                   and consumes NO random bits (so the tree does NOT blow up). The
#                   engine VERIFIES order-independence by re-BFSing each rep in a
#                   reversed and a shifted order and asserting an identical leaf
#                   multiset (the OIP cross-check). Result: the small tree of the
#                   sorted version AND the full species + point-group symmetry of the
#                   shuffled version.
#
# Expected result:  tau PASS,  DB PASS,  ERGODICITY PASS  (species S3, point group D4)
# ============================================================================

const NGRID          = 3
const MAXD2          = 2
const PARTICLE_TYPES = [1, 2, 3]

# Supplied direction set (the contract): the 8 unit displacements, closed under D4.
const MOVES = [(dx, dy) for dx in -1:1 for dy in -1:1 if (dx, dy) != (0, 0)]

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

wfwd_threshold(eInit::LinForm, eFwd::LinForm) =
    th_piece([([c_lt(eInit, eFwd)],
               th_sub(th_const(1), th_boltz(linsub(eFwd, eInit))))], th_const(0))

function pw_threshold(eInit::LinForm, eFwd::LinForm, eRev::LinForm)
    ratio = th_div(th_sub(th_const(1), th_boltz(linsub(eRev, eInit))),
                   th_sub(th_const(1), th_boltz(linsub(eFwd, eInit))))
    th_piece([([c_lt(eInit, eFwd), c_lt(eInit, eRev)], th_min(ratio, th_const(1))),
              ([c_lt(eInit, eFwd)], th_const(0))], th_const(0))
end

function build_cluster(rng, state::PState, n::Int, seedidx::Int, dir)
    cluster = [seedidx]; incluster = Set(cluster); queue = [seedidx]
    while !isempty(queue)
        pidx = popfirst!(queue); p = state[pidx]
        pPost = move(p, dir); pRev = move(p, rev(dir))
        cands = Int[]
        for qi in 1:length(state)
            (qi in incluster) && continue
            q = state[qi]
            dP = pbc_d2(q, p, n); dPost = pbc_d2(q, pPost, n); dRev = pbc_d2(q, pRev, n)
            ((0 < dP <= MAXD2) || (0 < dPost <= MAXD2) || (0 < dRev <= MAXD2)) && push!(cands, qi)
        end
        # ORDER-INDEPENDENT iteration: process the candidates in a canonical order
        # that reads no absolute label (species-/point-group-blind) and consumes no
        # random bits. Each per-candidate link/frustration decision below depends
        # only on (p, q, dir) and draws its own rng, so the cluster distribution is
        # independent of this order — the engine verifies that (the OIP cross-check).
        for qi in unordered(rng, cands)
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
    dir     = rand_move!(rng)

    cl = build_cluster(rng, state, NGRID, seedidx, dir)
    cl === :frustrated && return state

    clset = Set(cl)
    noncluster = eltype(state)[state[i] for i in 1:length(state) if !(i in clset)]
    for ci in cl
        dest = move(state[ci], dir)
        for q in noncluster
            same_site(q, dest, NGRID) && return state
        end
    end
    vcat(noncluster, eltype(state)[move(state[ci], dir) for ci in cl])
end
