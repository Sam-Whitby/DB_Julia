# ============================================================================
# hop_repeated_species.jl  —  species reduction with REPEATED multiplicities
# ============================================================================
# Four particles of two species with EQUAL multiplicities, [1,1,2,2], on a 3x3
# torus (zero energy, so pi is uniform). A correct symmetric 4-hop (choose a
# particle, choose one of the 4 neighbours uniformly, stay if occupied).
#
# The point of this example: the species-permutation group is the multiplicity-
# PRESERVING subgroup, here the single swap 1<->2 (it keeps two 1s and two 2s).
# The hop is type-blind, so it is species-equivariant and the type-taint certifies
# it: the tau-BFS is reduced over the combined (translation x {id, 1<->2}) orbits.
# This exercises the repeated-multiplicity path of _type_group_full / the combined
# reduction (relabelling that maps a state with duplicate labels to another valid
# state).
#
# Expected result:  tau PASS,  DB PASS,  ERGODICITY PASS,  species reduction USED.
# ============================================================================

const NGRID          = 3
const MAXD2          = 0              # zero energy: pi uniform
const PARTICLE_TYPES = [1, 1, 2, 2]

const HR_DISPS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

energy(state::PState)::LinForm = LinForm()

function algorithm(rng, state::PState)::PState
    pidx = rand_choice_index!(rng, length(state))
    p    = state[pidx]
    (dr, dc) = rand_choice!(rng, HR_DISPS)            # always 4 (no variable pool)
    newp = Particle(p.r + dr, p.c + dc, p.t)
    rest = state[setdiff(1:length(state), pidx)]
    for q in rest
        same_site(q, newp, NGRID) && return state      # occupied -> stay (symmetric)
    end
    vcat(rest, [newp])
end
