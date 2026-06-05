# ============================================================================
# broken_species_halfbeta.jl  —  species-DEPENDENT acceptance (taint declines)
# ============================================================================
# Single-particle Metropolis, but the acceptance exponent depends on the ABSOLUTE
# species of the moved particle: species 1 uses the correct exp(-beta*dE), while
# species 2 and 3 use the wrong exp(-beta*dE/2). This branches on the literal label
# (`p.t == 1`), so the algorithm is NOT species-equivariant.
#
# Two things must hold:
#   * the type-taint must DETECT the absolute-label branch and DECLINE the species
#     BFS reduction (species_free = false) — it must never wrongly relabel a
#     non-equivariant algorithm;
#   * detailed balance must still be checked correctly via the translation-only
#     fallback, and the dE/2 bug for species 2,3 must be CAUGHT (DB FAIL).
#
# Expected result:  tau PASS,  DB FAIL,  species reduction DECLINED.
# ============================================================================

const NGRID          = 3
const MAXD2          = 2
const PARTICLE_TYPES = [1, 2, 3]

const BS_DISPS = [(dx, dy) for dx in -1:1 for dy in -1:1 if (dx, dy) != (0, 0)]

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
    (dr, dc) = rand_choice!(rng, BS_DISPS)
    newp = Particle(p.r + dr, p.c + dc, p.t)
    rest = state[setdiff(1:length(state), pidx)]
    for q in rest
        same_site(q, newp, NGRID) && return state
    end
    newstate = vcat(rest, [newp])
    dE = linsub(energy(newstate), energy(state))
    # BUG + species dependence: species 1 uses dE, species 2/3 use dE/2.
    exponent = (p.t == 1) ? dE : LinForm(a => c * (1 // 2) for (a, c) in dE)
    metropolis!(rng, dE; exponent = exponent) ? newstate : state
end
