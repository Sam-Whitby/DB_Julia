# ============================================================================
# metropolis_4x4.jl  —  single-particle Metropolis on a 4x4 lattice (edge case)
# ============================================================================
# Same algorithm as single_metropolis but on a larger 4x4 torus with two
# interacting particles (240 states). Demonstrates that the checker handles
# lattices beyond 3x3 in a few seconds, and that interactions on a 4x4 geometry
# (more distinct squared distances) are verified correctly.
#
# Expected result:  tau PASS,  DB PASS,  ERGODICITY PASS
# ============================================================================

const NGRID          = 4
const MAXD2          = 4
const PARTICLE_TYPES = [1, 2]

const M4_DISPS = [(dx, dy) for dx in -1:1 for dy in -1:1 if (dx, dy) != (0, 0)]

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
    (dr, dc) = rand_choice!(rng, M4_DISPS)
    newp = Particle(p.r + dr, p.c + dc, p.t)
    rest = state[setdiff(1:length(state), pidx)]
    for q in rest
        same_site(q, newp, NGRID) && return state
    end
    newstate = vcat(rest, [newp])
    dE = linsub(energy(newstate), energy(state))
    metropolis!(rng, dE) ? newstate : state
end
