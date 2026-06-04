# ============================================================================
# broken_8way_hop.jl  —  Julia translation of examples/broken_8way_hop.wl
# ============================================================================
# BROKEN single-particle 8-hop (no Metropolis; energy = 0).  Same variable-pool
# defect as broken_variable_pool but in the k=3 bit bracket: pool size 7 or 8.
# Needs nGrid >= 4 so non-8-adjacent pairs exist (true probability 1/7 vs 1/8).
#
# Expected result:  tau PASS,  DB FAIL
# ============================================================================

const NGRID          = 4
const MAXD2          = 0
const PARTICLE_TYPES = [1, 2]

const B8_DISPS = [(dx, dy) for dx in -1:1 for dy in -1:1 if (dx, dy) != (0, 0)]

energy(state::PState)::LinForm = LinForm()

function algorithm(rng, state::PState)::PState
    pidx = rand_choice_index!(rng, length(state))
    p    = state[pidx]
    rest = state[setdiff(1:length(state), pidx)]

    valid = [(d1, d2) for (d1, d2) in B8_DISPS
             if !any(same_site(q, Particle(p.r + d1, p.c + d2, p.t), NGRID) for q in rest)]
    isempty(valid) && return state

    (d1, d2) = rand_choice!(rng, valid)        # BUG: pool 7 -> weight 1/7 != 1/8
    vcat(rest, [Particle(p.r + d1, p.c + d2, p.t)])
end
