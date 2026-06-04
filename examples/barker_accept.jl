# ============================================================================
# barker_accept.jl  —  single-particle moves with BARKER (Glauber) acceptance
# ============================================================================
# Identical proposal to single_metropolis, but the acceptance rule is Barker's
# instead of Metropolis':
#
#     A(s->t) = pi_t / (pi_s + pi_t) = 1 / (1 + exp(beta * dE)),   dE = E_t - E_s.
#
# Barker acceptance satisfies detailed balance:
#     A(s->t)/A(t->s) = (1+exp(-b dE))/(1+exp(b dE)) = exp(-b dE) = pi_t/pi_s.
#
# The weight on accept is 1/(1+exp(b dE)); on reject it is exp(b dE)/(1+exp(b dE)).
# The denominator (1 + exp(...)) is the new sign-definite-binomial case the engine
# now handles (it never vanishes, so clearing it is sound). This exercises the
# Tier-1 generalisation of `val_div` beyond the VMMC (1 - exp) denominators.
#
# Expected result:  tau PASS,  DB PASS,  ERGODICITY PASS
# ============================================================================

const NGRID          = 3
const MAXD2          = 2
const PARTICLE_TYPES = [1, 2, 3]

const BK_DISPS = [(dx, dy) for dx in -1:1 for dy in -1:1 if (dx, dy) != (0, 0)]

function energy(state::PState)::LinForm
    lf = LinForm()
    for i in 1:length(state), j in (i+1):length(state)
        d2 = pbc_d2(state[i], state[j], NGRID)
        (0 < d2 <= MAXD2) || continue
        addcoef!(lf, Jc(state[i].t, state[j].t, d2), TauNum(1))
    end
    lf
end

# Negate a linear form (build exp(+beta·X) as th_boltz(-X)).
neg(lf::LinForm) = LinForm(a => -c for (a, c) in lf)

# Barker acceptance threshold A = 1 / (1 + exp(beta*dE)).
barker_threshold(dE::LinForm) =
    th_div(th_const(1), th_add(th_const(1), th_boltz(neg(dE))))   # exp(beta*dE) = th_boltz(-dE)

function algorithm(rng, state::PState)::PState
    pidx = rand_choice_index!(rng, length(state))
    p    = state[pidx]
    (dr, dc) = rand_choice!(rng, BK_DISPS)
    newp = Particle(p.r + dr, p.c + dc, p.t)
    rest = state[setdiff(1:length(state), pidx)]
    for q in rest
        same_site(q, newp, NGRID) && return state
    end
    newstate = vcat(rest, [newp])
    dE = linsub(energy(newstate), energy(state))         # E_t - E_s
    accept!(rng, barker_threshold(dE)) ? newstate : state
end
