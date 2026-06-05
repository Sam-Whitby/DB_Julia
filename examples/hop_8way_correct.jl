# ============================================================================
# hop_8way_correct.jl  —  CORRECT 8-way hop on a 4x4 lattice (edge case)
# ============================================================================
# The detailed-balance-respecting counterpart of broken_8way_hop, and a larger
# (4x4, 240-state) system to exercise scaling. Zero energy, so pi is uniform and
# the transition matrix must be symmetric.
#
#   1. Choose a particle uniformly.
#   2. Choose one of the 8 neighbour displacements UNIFORMLY (always pool 8, a
#      power of two -> a clean k=3 bit draw with no rejection).
#   3. If the target is occupied, STAY (do not resample) — this keeps the move
#      symmetric: T(s->t)=1/(N*8)=T(t->s).
#
# This is the right way to handle hard-core exclusion (reject-and-stay), versus
# broken_*_hop which resamples from a variable-size pool and breaks symmetry.
#
# Expected result:  tau PASS,  DB PASS,  ERGODICITY PASS
# ============================================================================

const NGRID          = 4
const MAXD2          = 0              # zero energy: pi uniform
const PARTICLE_TYPES = [1, 2]

const MOVES = [(dx, dy) for dx in -1:1 for dy in -1:1 if (dx, dy) != (0, 0)]   # D4-closed

energy(state::PState)::LinForm = LinForm()

function algorithm(rng, state::PState)::PState
    pidx = rand_choice_index!(rng, length(state))
    p    = state[pidx]
    newp = move(p, rand_move!(rng))                     # ALWAYS 8 (no variable pool)
    rest = state[setdiff(1:length(state), pidx)]
    for q in rest
        same_site(q, newp, NGRID) && return state       # occupied -> stay (symmetric)
    end
    vcat(rest, [newp])
end
