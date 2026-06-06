# ============================================================================
# vmmc_early_stop.jl  —  VMMC with EARLY cluster-growth termination
# ============================================================================
# A common-sounding VMMC variant: the cluster growth is stopped EARLY with some
# probability (here 1/2 after each fully-processed cluster particle), and the
# partial cluster is moved. This file is a stress test on TWO axes at once:
#
#  1. ORDER-INDEPENDENCE (does the `unordered` OIP cross-check behave correctly?).
#     The early stop happens AFTER a cluster particle's whole candidate loop, so the
#     cluster reached is the same SET regardless of the order the candidates were
#     visited in. The move is therefore order-INDEPENDENT, and the checker correctly
#     ACCEPTS the `unordered` reduction (species + point group). Contrast this with a
#     stop placed *inside* the candidate loop, which makes the partial cluster depend
#     on visiting order — that the cross-check correctly REJECTS (see test_db.jl).
#
#  2. CORRECTNESS. Terminating the cluster early WITHOUT compensating the acceptance
#     breaks the Whitelam–Geissler construction: the move no longer samples the
#     Boltzmann distribution. The checker catches this — it reports detailed balance
#     FAIL *and* global balance (`-balance`) FAIL. So this is a faithfully-translated
#     but PHYSICALLY WRONG algorithm, exactly the kind an LLM-in-the-loop search must
#     have rejected by the code-checker.
#
# Expected result:  tau PASS,  detailed balance FAIL,  global balance FAIL
#                   (OIP cross-check PASSES — the move is order-independent)
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
        for qi in unordered(rng, cands)              # order-independent (set is fixed)
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
        # EARLY STOP (after a whole particle's candidates -> order-independent): with
        # probability 1/2, stop growing and move the partial cluster. This is what
        # breaks detailed balance and global balance (no compensating acceptance).
        (!isempty(queue) && accept!(rng, th_const(1//2))) && break
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
