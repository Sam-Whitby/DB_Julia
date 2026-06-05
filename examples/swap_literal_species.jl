# ============================================================================
# swap_literal_species.jl  —  a swap that re-writes RAW labels (covariance guard)
# ============================================================================
# Identical in OUTCOME to kawasaki (pick a distinct-species pair, swap their
# species, accept with Metropolis), but the new species labels are written as RAW
# integers extracted with `typeval` rather than carried through as the inherited
# tags. The transition probabilities — and hence detailed balance — are exactly
# kawasaki's, so DB PASSES.
#
# The point: an output label that is a fresh integer (not an inherited tag) would
# NOT relabel under a species permutation, so the move is not leaf-level species-
# equivariant. The species-COVARIANCE guard detects this (an output particle whose
# `.t` is not a TypeTag) and conservatively DECLINES the species BFS reduction —
# falling back to the (still correct) translation reduction. This is the species
# analogue of the position-covariance guard, and it shows the checker never trusts
# an output it cannot certify equivariant.
#
# Expected result:  tau PASS,  DB PASS,  ERGODICITY FAIL (only N! perms reachable),
#                   species reduction DECLINED.
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
    pairs = [(i, j) for i in 1:length(state) for j in (i+1):length(state)
             if state[i].t != state[j].t]               # `!=` between labels: allowed
    isempty(pairs) && return state

    (i, j) = rand_choice!(rng, pairs)
    ti, tj = typeval(state[i].t), typeval(state[j].t)    # extract RAW labels (not tags)
    newstate = [k == i ? Particle(state[k].r, state[k].c, tj) :   # write raw Int -> not a tag
                k == j ? Particle(state[k].r, state[k].c, ti) :
                state[k] for k in 1:length(state)]

    dE = linsub(energy(newstate), energy(state))
    metropolis!(rng, dE) ? newstate : state
end
