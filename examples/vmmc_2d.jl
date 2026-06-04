# ============================================================================
# vmmc_2d.jl  —  Julia translation of examples/vmmc_2d.wl
# ============================================================================
# Virtual-move Monte Carlo (Whitelam-Geissler cluster algorithm) on a 2D torus.
#   1. Choose a seed particle and a displacement uniformly.
#   2. Build a cluster: for each cluster particle p and each occupied neighbour
#      q (within maxD2 of p, p+dir, or p-dir, in canonical order), draw r1 and
#      link q forward with probability wFwd; if linked, draw r2 and either keep
#      the link or declare the move "frustrated" (whole move rejected).
#   3. Translate the cluster by dir; reject on hard-core overlap.
#
# FAITHFUL translation: the two-RandomReal (r1, r2) frustration test is mirrored
# exactly, including the Min[ratio, 1] threshold whose ratio
#   (1 - exp(beta*(eInit-eRev))) / (1 - exp(beta*(eInit-eFwd)))
# cancels only after the r1 and r2 factors are multiplied. The checker carries
# leaf weights as exact rational functions of the exponentials (num / product of
# (1-exp) binomials) and clears denominators in the detailed-balance residual.
#
# Expected result:  tau PASS,  DB PASS  (VMMC satisfies detailed balance)
# ============================================================================

const NGRID          = 3
const MAXD2          = 2
const PARTICLE_TYPES = [1, 2, 3]

# physLen = 1  =>  nStep = 1  =>  the 8 unit displacements.
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

# Pairwise interaction energy between types ti,tj at positions pi,pj (a LinForm:
# one coupling atom if within range, else 0).
function pairE(ti::Int, tj::Int, pi::Particle, pj::Particle, n::Int)::LinForm
    lf = LinForm()
    d2 = pbc_d2(pi, pj, n)
    (0 < d2 <= MAXD2) && addcoef!(lf, Jc(ti, tj, d2), TauNum(1))
    lf
end

shift(p::Particle, d) = Particle(p.r + d[1], p.c + d[2], p.t)

# wFwd threshold = (eInit<eFwd) ? 1 - exp(beta*(eInit-eFwd)) : 0
wfwd_threshold(eInit::LinForm, eFwd::LinForm) =
    th_piece([([c_lt(eInit, eFwd)],
               th_sub(th_const(1), th_boltz(linsub(eFwd, eInit))))], th_const(0))

# Pw threshold (frustration acceptance), mirroring the .wl Piecewise:
#   {Min[(1-exp(b(eInit-eRev)))/(1-exp(b(eInit-eFwd))), 1], eInit<eFwd && eInit<eRev}
#   {0, eInit<eFwd}, default 0
function pw_threshold(eInit::LinForm, eFwd::LinForm, eRev::LinForm)
    ratio = th_div(th_sub(th_const(1), th_boltz(linsub(eRev, eInit))),
                   th_sub(th_const(1), th_boltz(linsub(eFwd, eInit))))
    th_piece([([c_lt(eInit, eFwd), c_lt(eInit, eRev)], th_min(ratio, th_const(1))),
              ([c_lt(eInit, eFwd)], th_const(0))], th_const(0))
end

# Whitelam-Geissler cluster builder. Returns the cluster (vector of state
# indices) or :frustrated. Cluster membership is tracked by particle index.
function build_cluster(rng, state::PState, n::Int, seedidx::Int, dir)
    cluster = [seedidx]; incluster = Set(cluster); queue = [seedidx]
    while !isempty(queue)
        pidx = popfirst!(queue); p = state[pidx]
        pPost = shift(p, dir); pRev = shift(p, (-dir[1], -dir[2]))
        # Candidate occupied neighbours not yet in the cluster.
        cands = Int[]
        for qi in 1:length(state)
            (qi in incluster) && continue
            q = state[qi]
            dP = pbc_d2(q, p, n); dPost = pbc_d2(q, pPost, n); dRev = pbc_d2(q, pRev, n)
            ((0 < dP <= MAXD2) || (0 < dPost <= MAXD2) || (0 < dRev <= MAXD2)) && push!(cands, qi)
        end
        # Canonical ordering by distance tuples (tau-free), matching the .wl.
        sort!(cands; by = qi -> (pbc_d2(state[qi], p, n), pbc_d2(state[qi], pPost, n),
                                 pbc_d2(state[qi], pRev, n), state[qi].t))
        for qi in cands
            q = state[qi]
            eInit = pairE(p.t, q.t, p,     q, n)
            eFwd  = pairE(p.t, q.t, pPost, q, n)
            eRev  = pairE(p.t, q.t, pRev,  q, n)
            if accept!(rng, wfwd_threshold(eInit, eFwd))          # r1 <= wFwd
                if !accept!(rng, pw_threshold(eInit, eFwd, eRev)) # r2 > Pw  => frustrated
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
    noncluster = Particle[state[i] for i in 1:length(state) if !(i in clset)]

    # Hard-core overlap: a moved cluster particle must not land on a non-cluster site.
    for ci in cl
        dest = shift(state[ci], dir)
        for q in noncluster
            same_site(q, dest, NGRID) && return state
        end
    end
    vcat(noncluster, Particle[shift(state[ci], dir) for ci in cl])
end
