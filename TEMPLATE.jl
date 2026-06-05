# ============================================================================
# TEMPLATE.jl  —  how to write an algorithm file for DB_Julia
# ============================================================================
# Copy this file, rename it, and fill in the two functions. An algorithm file is
# a Julia translation of ONE Monte-Carlo step. It must define exactly these five
# top-level names (the checker reads them by name):
#
#     NGRID           :: Int            lattice side length (n x n torus)
#     MAXD2           :: Int            largest squared interaction distance kept
#     PARTICLE_TYPES  :: Vector{Int}    the multiset of particle "species"
#     energy(state)   :: LinForm        the system's energy, symbolic in couplings
#     algorithm(rng, state) :: PState   one MCMC step, using the primitives below
#
# THE GOLDEN RULE
# ---------------
# A particle's coordinates (`p.r`, `p.c`) are special numbers (`TauNum`) that
# secretly carry the lattice-translation offset, so the checker can prove
# translational invariance for free. You never see or touch that offset. Just:
#
#   * do ordinary arithmetic on coordinates  (`p.r + dr`, `p.c - 1`, ...);
#   * ask geometric questions ONLY through `pbc_d2`, `same_site`, `pmod`;
#   * NEVER compare coordinates with `<`, and never convert one to a plain Int
#     yourself.  Those are the only ways to leak an absolute position into a
#     decision, and the checker forbids them (it errors, it does not guess).
#
# Follow that rule and the checker's verdict is exact and fully trustworthy. You
# do NOT need to do anything to "declare" translational invariance — it is always
# measured and reported, and getting it wrong does not by itself fail your run.
# ----------------------------------------------------------------------------

# ---- 1. System size and species -------------------------------------------
const NGRID          = 3              # n x n periodic lattice
const MAXD2          = 2              # keep pair interactions with 0 < d^2 <= 2
const PARTICLE_TYPES = [1, 2, 3]      # three distinct particles (use repeats for
                                      # identical species, e.g. [1, 1, 2])

# A reusable list of displacements (here the 8 nearest/next-nearest neighbours).
const DISPS = [(dx, dy) for dx in -1:1 for dy in -1:1 if (dx, dy) != (0, 0)]

# ---- 2. Energy -------------------------------------------------------------
# `energy(state)` returns a LinForm: a symbolic linear combination of coupling
# parameters. You build it by adding one term per interacting pair. The checker
# verifies detailed balance for ALL values of these couplings at once.
#
#   Jc(t1, t2, d2)        the pair coupling between species t1, t2 at squared
#                         distance d2  (written couplingJ[t1,t2,d2])
#   Xparam(:fieldH)       a scalar field-like parameter named :fieldH
#   addcoef!(lf, atom, c) add coefficient c (a TauNum, usually TauNum(1)) of atom
#
# `state` is a Vector{Particle}; each Particle has `.r`, `.c` (coordinates) and
# `.t` (its integer species).
function energy(state::PState)::LinForm
    lf = LinForm()
    for i in 1:length(state), j in (i+1):length(state)
        d2 = pbc_d2(state[i], state[j], NGRID)        # squared min-image distance
        (0 < d2 <= MAXD2) || continue
        addcoef!(lf, Jc(state[i].t, state[j].t, d2), TauNum(1))
    end
    lf
end

# ---- 3. One MCMC step ------------------------------------------------------
# `algorithm(rng, state)` performs one step and returns the next state. Use these
# random primitives — and ONLY these — for every random choice (the checker
# replays each one exactly over its whole decision tree):
#
#   rand_choice_index!(rng, n)   uniform index in 1..n          (weight 1/n)
#   rand_choice!(rng, list)      uniform element of `list`       (weight 1/len)
#   rand_integer!(rng, lo, hi)   uniform integer in [lo,hi]      (weight 1/(hi-lo+1))
#   metropolis!(rng, dE)         accept with min(1, exp(-beta*dE)); dE::LinForm
#   accept!(rng, thr)            accept with a custom symbolic threshold thr
#   unordered(rng, list)         iterate `list` in an order-independent way (OIP)
#
# THE ORDER-INDEPENDENT PRIMITIVE (`unordered`)
# ---------------------------------------------
# If your move processes a set of candidates in a loop whose RESULT does not depend
# on the visiting order (e.g. a cluster builder where each candidate's link decision
# is independent), write `for q in unordered(rng, cands)`. Two reasons:
#   * it reads NONE of the items' content, so it cannot break a species / point-group
#     symmetry the way a sort tie-breaking on `state[qi].t` would (that tie-break is
#     exactly why vmmc_2d.jl declines species);
#   * it consumes NO random bits, so it does NOT blow up the decision tree the way an
#     explicit Fisher-Yates shuffle does (which adds a factor of |cands|! per step).
# You are ASSERTING order-independence; the checker VERIFIES it (it re-BFSes each
# representative in a second candidate order and checks the transition probabilities
# are identical) and raises a HARD ERROR if your body actually depends on order.
# See examples/vmmc_2d_unordered.jl.
#
# For custom thresholds (cluster algorithms etc.) build `thr` with th_const,
# th_boltz, th_sub, th_div, th_min, th_max, th_piece, c_lt, c_le — see vmmc_2d.jl.
# th_div may divide by 1+exp as well as 1-exp (Barker/Glauber — see barker_accept.jl);
# th_linear(L) is the bare value <L,J>, so a weight can carry a coupling/rate factor
# (see poly_rate_accept.jl).
#
# Build a moved particle with plain arithmetic: `Particle(p.r + dr, p.c + dc, p.t)`.
# Do NOT wrap coordinates into the box yourself; the checker normalises the
# returned state onto the torus for you.
function algorithm(rng, state::PState)::PState
    pidx = rand_choice_index!(rng, length(state))      # pick a particle
    p    = state[pidx]
    (dr, dc) = rand_choice!(rng, DISPS)                # pick a displacement
    newp = Particle(p.r + dr, p.c + dc, p.t)
    rest = state[setdiff(1:length(state), pidx)]

    # Hard-core exclusion: reject if the target site is occupied. `same_site`
    # compares positions safely (it works on the difference, which is absolute-
    # position-free).
    for q in rest
        same_site(q, newp, NGRID) && return state
    end

    newstate = vcat(rest, [newp])
    dE = linsub(energy(newstate), energy(state))       # energy(new) - energy(old)
    metropolis!(rng, dE) ? newstate : state            # accept or stay
end

# ----------------------------------------------------------------------------
# Run it:   julia --project=. check.jl your_file.jl
# Optional: -maxdepth N   (raise the path-length cap; default 30)
#           -parallel     (use multiple CPU cores; needs `julia -t auto`)
#           -balance      (check global balance pi*T=pi instead of the stronger
#                          detailed balance — accepts correct non-reversible moves)
# ----------------------------------------------------------------------------
