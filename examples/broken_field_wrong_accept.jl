# ============================================================================
# broken_field_wrong_accept.jl  —  Julia translation of examples/broken_field_wrong_accept.wl
# ============================================================================
# The DECLARED energy includes a quadratic field (pairwise + fieldH*sum(row^2)),
# but the algorithm's Metropolis acceptance uses ONLY the pairwise dE, ignoring
# the field.  The acceptance is therefore tau-free (tau PASS), but the sampled
# distribution does not match the full-energy Boltzmann weight, so DB FAILS.
#
# Expected result:  tau PASS,  DB FAIL
# ============================================================================

const NGRID          = 2
const MAXD2          = 2
const PARTICLE_TYPES = [1, 2]

const BF_DISPS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

# Pairwise-only energy (used by the algorithm's acceptance — tau-free).
function pair_energy(state::PState)::LinForm
    lf = LinForm()
    for i in 1:length(state), j in (i+1):length(state)
        d2 = pbc_d2(state[i], state[j], NGRID)
        (0 < d2 <= MAXD2) || continue
        addcoef!(lf, Jc(state[i].t, state[j].t, d2), TauNum(1))
    end
    lf
end

# Full declared energy (used by the DB check): pairwise + quadratic field.
function energy(state::PState)::LinForm
    lf = pair_energy(state)
    fld = TauNum(0)
    for p in state; fld = fld + p.r^2; end
    addcoef!(lf, Xparam(:fieldH), fld)
    lf
end

function algorithm(rng, state::PState)::PState
    pidx = rand_choice_index!(rng, length(state))
    p    = state[pidx]
    (dr, dc) = rand_choice!(rng, BF_DISPS)
    newp = Particle(p.r + dr, p.c + dc, p.t)
    rest = state[setdiff(1:length(state), pidx)]

    for q in rest
        same_site(q, newp, NGRID) && return state
    end

    newstate = vcat(rest, [newp])
    # BUG: acceptance uses ONLY the pairwise dE, ignoring the field.
    dE = linsub(pair_energy(newstate), pair_energy(state))
    metropolis!(rng, dE) ? newstate : state
end
