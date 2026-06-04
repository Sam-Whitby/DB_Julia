# ============================================================================
# poly_rate_accept.jl  —  rate-limited Metropolis (polynomial weight factor)
# ============================================================================
# single_metropolis, but every proposed move must ALSO pass an extra coin that
# succeeds with probability equal to a free "rate" parameter `a` (a bare coupling,
# not a Boltzmann factor). The transition weight to a new state is therefore
#
#     (1/N)(1/D) * a * min(1, exp(-beta*dE))
#
# i.e. a POLYNOMIAL (degree-1, the factor `a`) multiplied by an exp-monomial. This
# exercises the Tier-2 polynomial-coefficient ring: weight coefficients are now
# polynomials in the couplings, not just rationals.
#
# Detailed balance still holds: the common factor `a` cancels between forward and
# reverse, leaving the ordinary Metropolis identity. The checker must verify a
# residual that is a polynomial in `a` times exp-monomials, and confirm every such
# polynomial coefficient is identically zero. (`a` ranges over all reals; DB holds
# for every value, including the physical 0<=a<=1.)
#
# Expected result:  tau PASS,  DB PASS,  ERGODICITY PASS
# ============================================================================

const NGRID          = 3
const MAXD2          = 2
const PARTICLE_TYPES = [1, 2, 3]

const PR_DISPS = [(dx, dy) for dx in -1:1 for dy in -1:1 if (dx, dy) != (0, 0)]

function energy(state::PState)::LinForm
    lf = LinForm()
    for i in 1:length(state), j in (i+1):length(state)
        d2 = pbc_d2(state[i], state[j], NGRID)
        (0 < d2 <= MAXD2) || continue
        addcoef!(lf, Jc(state[i].t, state[j].t, d2), TauNum(1))
    end
    lf
end

# The acceptance probability is the bare rate parameter a (a polynomial weight).
rate_threshold() = th_linear(LinForm(Xparam(:a) => TauNum(1)))

function algorithm(rng, state::PState)::PState
    pidx = rand_choice_index!(rng, length(state))
    p    = state[pidx]
    (dr, dc) = rand_choice!(rng, PR_DISPS)
    newp = Particle(p.r + dr, p.c + dc, p.t)
    rest = state[setdiff(1:length(state), pidx)]
    for q in rest
        same_site(q, newp, NGRID) && return state
    end
    accept!(rng, rate_threshold()) || return state       # rate coin: prob a
    newstate = vcat(rest, [newp])
    dE = linsub(energy(newstate), energy(state))
    metropolis!(rng, dE) ? newstate : state              # then ordinary Metropolis
end
