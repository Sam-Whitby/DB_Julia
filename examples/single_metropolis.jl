# ============================================================================
# single_metropolis.jl  —  Julia translation of examples/single_metropolis.wl
# ============================================================================
# Single-particle Metropolis on a 2D periodic lattice.
#   1. Choose one particle uniformly.
#   2. Choose a displacement uniformly from the 8 nearest/next-nearest dirs.
#   3. Reject immediately if the target site is occupied.
#   4. Accept with probability min(1, exp(-beta * dE)).
#
# Expected result:  tau PASS,  DB PASS
# ============================================================================

const NGRID          = 3
const MAXD2          = 2
const PARTICLE_TYPES = [1, 2, 3]

const SM_DISPS = [(dx, dy) for dx in -1:1 for dy in -1:1 if (dx, dy) != (0, 0)]

# Pairwise energy: sum of couplingJ[type_a, type_b, d2] over close pairs.
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
    (dr, dc) = rand_choice!(rng, SM_DISPS)
    newp = Particle(p.r + dr, p.c + dc, p.t)      # no Mod — harness normalises
    rest = state[setdiff(1:length(state), pidx)]

    # Hard-core rejection: target occupied?  (difference => tau cancels)
    for q in rest
        same_site(q, newp, NGRID) && return state
    end

    newstate = vcat(rest, [newp])
    dE = linsub(energy(newstate), energy(state))
    metropolis!(rng, dE) ? newstate : state
end
