# ============================================================================
# broken_biased_direction.jl  —  Julia translation of examples/broken_biased_direction.wl
# ============================================================================
# Identical to single_metropolis except the displacement (0,1) ("right")
# appears TWICE in the proposal list, so it is chosen with probability 2/9 while
# every other direction has 1/9.  The reverse move ("left") still has 1/9, so
# T(s->t)/T(t->s) = 2 * pi(t)/pi(s) for every rightward move: DB is violated.
#
# Expected result:  tau PASS,  DB FAIL
# ============================================================================

const NGRID          = 3
const MAXD2          = 2
const PARTICLE_TYPES = [1, 2, 3]

# (0,1) appears twice => P(right) = 2/9.
const BD_DISPS = [(0,1), (0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]

function energy(state::PState)::LinForm
    lf = LinForm()
    for i in 1:length(state), j in (i+1):length(state)
        d2 = pbc_d2(state[i], state[j], NGRID)
        (0 < d2 <= MAXD2) || continue
        addcoef!(lf, Jc(state[i].t, state[j].t, d2), TauNum(1))
    end
    lf
end

function algorithm(rng, state::PState)::PState
    pidx = rand_choice_index!(rng, length(state))
    p    = state[pidx]
    (dr, dc) = rand_choice!(rng, BD_DISPS)        # biased: right appears twice
    newp = Particle(p.r + dr, p.c + dc, p.t)
    rest = state[setdiff(1:length(state), pidx)]

    for q in rest
        same_site(q, newp, NGRID) && return state
    end

    newstate = vcat(rest, [newp])
    dE = linsub(energy(newstate), energy(state))
    metropolis!(rng, dE) ? newstate : state
end
