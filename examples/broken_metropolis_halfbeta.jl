# ============================================================================
# broken_metropolis_halfbeta.jl  —  Julia translation of examples/broken_metropolis_halfbeta.wl
# ============================================================================
# Identical to single_metropolis except the acceptance uses exp(-beta*dE/2)
# instead of exp(-beta*dE).  The accept/reject ratio is then exp(-beta*dE/2),
# not pi(t)/pi(s) = exp(-beta*dE), so detailed balance is violated.
#
# This produces HALF-INTEGER Boltzmann exponents.  The Julia checker groups by
# EXACT rational exponent vectors, so it verifies this case directly (the
# Mathematica path falls back here; pure-Julia does not need to).
#
# Expected result:  tau PASS,  DB FAIL
# ============================================================================

const NGRID          = 3
const MAXD2          = 2
const PARTICLE_TYPES = [1, 2, 3]

const MH_DISPS = [(dx, dy) for dx in -1:1 for dy in -1:1 if (dx, dy) != (0, 0)]

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
    (dr, dc) = rand_choice!(rng, MH_DISPS)
    newp = Particle(p.r + dr, p.c + dc, p.t)
    rest = state[setdiff(1:length(state), pidx)]

    for q in rest
        same_site(q, newp, NGRID) && return state
    end

    newstate = vcat(rest, [newp])
    dE   = linsub(energy(newstate), energy(state))
    # BUG: exponent uses dE/2 instead of dE (clamp condition is still dE <= 0).
    half = LinForm(a => c * (1 // 2) for (a, c) in dE)
    metropolis!(rng, dE; exponent = half) ? newstate : state
end
