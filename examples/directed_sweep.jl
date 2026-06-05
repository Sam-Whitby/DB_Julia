# ============================================================================
# directed_sweep.jl  —  a NON-REVERSIBLE chain: balance PASS, detailed balance FAIL
# ============================================================================
# The minimal demonstration that GLOBAL BALANCE (pi*T = pi, what correct sampling
# actually requires) is strictly WEAKER than DETAILED BALANCE (the sufficient
# condition the default check verifies).
#
# A single particle is deterministically shifted one column forward every step.
# The transition matrix is a cyclic PERMUTATION of the states, hence doubly
# stochastic, so the uniform distribution (here pi is uniform: zero energy) is
# stationary -> GLOBAL BALANCE HOLDS. But the move is directed: T(s->t) = 1 while
# the reverse T(t->s) = 0, so DETAILED BALANCE FAILS for every moved pair. This is
# the same reason event-chain / lifting / Suwa-Todo samplers are correct yet not
# reversible.
#
# Run the default checker  ->  Detailed balance: FAIL.
# Run with -balance        ->  Balance: PASS.
#
#   julia --project=. check.jl examples/directed_sweep.jl            # FAIL (DB)
#   julia --project=. check.jl examples/directed_sweep.jl -balance   # PASS (balance)
#
# Ergodicity FAILS by design: a single forward shift visits only one row's cycle,
# not the whole torus (a translation has order n, so it cannot mix all n^2 states;
# genuine non-reversible *ergodic* mixing needs a lifted direction d.o.f., which is
# beyond a single covariant move).
#
# Expected result:  tau PASS,  detailed balance FAIL,  balance PASS,  ergodicity FAIL
# ============================================================================

const NGRID          = 3
const MAXD2          = 0           # zero energy: pi uniform
const PARTICLE_TYPES = [1]         # a single particle

energy(state::PState)::LinForm = LinForm()    # zero energy -> uniform target

function algorithm(rng, state::PState)::PState
    p = state[1]
    # Deterministic directed move: always +1 column. Covariant (shifts with the
    # lattice), so translational invariance still holds; it is the DIRECTION (no
    # reverse move is ever proposed) that breaks detailed balance while preserving
    # the uniform stationary distribution.
    [Particle(p.r, p.c + 1, p.t)]
end
