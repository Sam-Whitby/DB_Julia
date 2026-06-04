# ============================================================================
# broken_variable_pool.jl  —  Julia translation of examples/broken_variable_pool.wl
# ============================================================================
# BROKEN single-particle 4-hop (no Metropolis; energy = 0, pi uniform).
#   1. Choose a particle uniformly.
#   2. Collect unoccupied 4-neighbours (pool size 3 or 4).
#   3. Hop to one chosen uniformly.
#
# When the two particles are adjacent the pool is 3, otherwise 4.  The true hop
# probability (1/3 vs 1/4) is therefore asymmetric between a state and its
# reverse, violating detailed balance.  The checker computes the exact 1/3 and
# 1/4 selection weights and detects the asymmetry.
#
# Expected result:  tau PASS,  DB FAIL
# ============================================================================

const NGRID          = 3
const MAXD2          = 0          # zero energy: pi uniform, T must be symmetric
const PARTICLE_TYPES = [1, 2]

const BVP_DISPS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

energy(state::PState)::LinForm = LinForm()   # zero energy

function algorithm(rng, state::PState)::PState
    pidx = rand_choice_index!(rng, length(state))
    p    = state[pidx]
    rest = state[setdiff(1:length(state), pidx)]

    valid = [(d1, d2) for (d1, d2) in BVP_DISPS
             if !any(same_site(q, Particle(p.r + d1, p.c + d2, p.t), NGRID) for q in rest)]
    isempty(valid) && return state

    (d1, d2) = rand_choice!(rng, valid)        # BUG: pool 3 -> weight 1/3 != 1/4
    vcat(rest, [Particle(p.r + d1, p.c + d2, p.t)])
end
