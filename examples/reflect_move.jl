# ============================================================================
# reflect_move.jl  —  a NON-translation-covariant move (soundness edge case)
# ============================================================================
# Picks a particle and reflects its row about the lattice origin (r -> -r mod n),
# leaving the column unchanged. This is a deterministic involution per particle,
# so with zero energy it actually SATISFIES detailed balance — but it is NOT
# translation invariant: the next-state position depends on the absolute row.
#
# This is the case that the orbit-reduction optimisation must NOT be applied to:
# the transition matrix is not translation-equivariant, so translating one
# representative's leaves to the rest of its orbit would give WRONG transitions
# and could yield a wrong verdict. The checker's covariance guard detects the
# non-covariant output position (its tau-row coefficient is -1, not +1), reports
# translational FAIL, and falls back to a direct BFS from every state — which is
# correct for any algorithm. The DB verdict is therefore still trustworthy.
#
# Expected result:  tau FAIL,  DB PASS  (verified via the all-states fallback)
# ============================================================================

const NGRID          = 3
const MAXD2          = 0              # zero energy: pi uniform
const PARTICLE_TYPES = [1, 2]

energy(state::PState)::LinForm = LinForm()

function algorithm(rng, state::PState)::PState
    pidx = rand_choice_index!(rng, length(state))
    p    = state[pidx]
    newp = Particle(-p.r, p.c, p.t)                 # reflect row: NOT covariant
    rest = state[setdiff(1:length(state), pidx)]
    for q in rest
        same_site(q, newp, NGRID) && return state    # keep it an involution
    end
    vcat(rest, [newp])
end
