# ============================================================================
# kawasaki.jl  —  Julia translation of examples/kawasaki.wl
# ============================================================================
# Kawasaki exchange dynamics: pick a pair of distinct-type particles, swap
# their types (positions unchanged), accept with min(1, exp(-beta*dE)).
#
# Only the N! = 6 type-permutations of the seed's positions are reachable, so
# ergodicity over the full 504-state space FAILS by design (correct physics).
#
# Expected result:  tau PASS,  DB PASS,  ERGODICITY FAIL (by design)
# ============================================================================

const NGRID          = 3
const MAXD2          = 2
const PARTICLE_TYPES = [1, 2, 3]

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
    # All index pairs (i<j) with distinct types.
    pairs = [(i, j) for i in 1:length(state) for j in (i+1):length(state)
             if state[i].t != state[j].t]
    isempty(pairs) && return state

    (i, j) = rand_choice!(rng, pairs)
    ti, tj = state[i].t, state[j].t
    newstate = [k == i ? Particle(state[k].r, state[k].c, tj) :
                k == j ? Particle(state[k].r, state[k].c, ti) :
                state[k] for k in 1:length(state)]

    dE = linsub(energy(newstate), energy(state))
    metropolis!(rng, dE) ? newstate : state
end
