# ============================================================================
# horizontal_metropolis.jl  —  single-particle Metropolis, horizontal moves only
# ============================================================================
# Displacement set {(0,+1), (0,−1)}: particles can only change their COLUMN.
# Rows never change, so not all states are reachable from the seed (ERGODICITY FAIL
# by design), but the Metropolis acceptance is still exactly correct, so DB PASS.
#
# This example illustrates that the symmetry-reduction step correctly identifies
# D2 symmetry (NOT the full D4):
#
#   VERIFIED:   rotate180, reflect_h, reflect_v  (D2 point group + translations)
#   NOT VERIFIED: rotate90, rotate270, reflect (diagonal), reflect_ad
#
# Reason: a 90-degree rotation maps a column-change displacement to a row-change
# displacement, which is outside the proposal set. The 180-degree rotation maps
# (0,+1)↔(0,−1) and (0,−1)↔(0,+1), both of which remain in the proposal set.
# The horizontal-axis and vertical-axis reflections similarly preserve the proposal
# set. Diagonal reflections exchange rows and columns and so fail.
#
# This is also a concrete example that point-group symmetry and detailed balance
# are INDEPENDENT: this algorithm has D4 spatial anisotropy (D2 ≠ D4) yet still
# satisfies detailed balance -- showing that a D4-FAIL verdict on the graph can
# NEVER be used to conclude DB-FAIL (the biased-direction examples break BOTH,
# but only because of the specific form of their defect, not as a general rule).
#
# Expected result:  tau PASS,  DB PASS,  ERGODICITY FAIL (rows fixed by design)
# ============================================================================

const NGRID          = 3
const MAXD2          = 2
const PARTICLE_TYPES = [1, 2]

# Only horizontal (column) moves. (0,+1) = move right; (0,−1) = move left.
# This set is closed under D2 = {identity, rotate180, reflect_h, reflect_v} but NOT
# under rotate90 (which maps a column move to a row move, outside the set), so the
# point-group reduction discovers exactly D2 — see the header note above.
const MOVES = [(0, 1), (0, -1)]

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
    newp = move(p, rand_move!(rng))                     # only column changes
    rest = state[setdiff(1:length(state), pidx)]
    for q in rest
        same_site(q, newp, NGRID) && return state
    end
    newstate = vcat(rest, [newp])
    dE = linsub(energy(newstate), energy(state))
    metropolis!(rng, dE) ? newstate : state
end
