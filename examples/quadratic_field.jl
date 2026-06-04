# ============================================================================
# quadratic_field.jl  —  Julia translation of examples/quadratic_field.wl
# ============================================================================
# Single-particle Metropolis with a QUADRATIC external field:
#     energy = pairwise couplingJ + fieldH * sum(row_i^2)
#
# The quadratic field makes dE depend on the particle's ABSOLUTE row, so
# translational invariance FAILS.  The algorithm also normalises newPos with an
# in-body Mod on the absolute position (required so the field uses the wrapped
# row), which is itself a tau-dependent operation.  Both are detected.
#
# The DB check uses the full declared energy (field included) and the algorithm
# accepts with the full dE, so detailed balance PASSES.
#
# Expected result:  tau FAIL,  DB PASS
# ============================================================================

const NGRID          = 2
const MAXD2          = 2
const PARTICLE_TYPES = [1, 2]

const QF_DISPS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

# Full energy: pairwise + quadratic field.  row_i^2 on a tau-augmented row is
# tau-dependent (tainted), which is how the field is detected as non-invariant.
function energy(state::PState)::LinForm
    lf = LinForm()
    for i in 1:length(state), j in (i+1):length(state)
        d2 = pbc_d2(state[i], state[j], NGRID)
        (0 < d2 <= MAXD2) || continue
        addcoef!(lf, Jc(state[i].t, state[j].t, d2), TauNum(1))
    end
    fld = TauNum(0)
    for p in state; fld = fld + p.r^2; end
    addcoef!(lf, Xparam(:fieldH), fld)
    lf
end

function algorithm(rng, state::PState)::PState
    pidx = rand_choice_index!(rng, length(state))
    p    = state[pidx]
    (dr, dc) = rand_choice!(rng, QF_DISPS)
    newp = Particle(p.r + dr, p.c + dc, p.t)
    rest = state[setdiff(1:length(state), pidx)]

    for q in rest
        same_site(q, newp, NGRID) && return state
    end

    # Normalise to canonical grid coordinates before computing energy.  pmod on
    # an absolute (tau-dependent) position flags the translational violation.
    nr = pmod(newp.r - 1, NGRID) + 1
    nc = pmod(newp.c - 1, NGRID) + 1
    newpn = Particle(TauNum(nr), TauNum(nc), p.t)

    newstate = vcat(rest, [newpn])
    dE = linsub(energy(newstate), energy(state))
    metropolis!(rng, dE) ? newstate : state
end
