#!/usr/bin/env julia
# ============================================================================
# check.jl  —  DB_Julia command-line checker
# ============================================================================
# Usage:
#     julia --project=. check.jl examples/<algorithm>.jl [-maxdepth N]
#
# The algorithm file (a Julia translation of one MCMC step) must define, as
# plain top-level bindings:
#
#     const NGRID          ::Int            # lattice side length
#     const MAXD2          ::Int            # max squared interaction distance
#     const PARTICLE_TYPES ::Vector{Int}    # type multiset
#     energy(state::PState)        ::LinForm        # symbolic energy (linear in couplings)
#     algorithm(rng, state::PState)::PState         # one MCMC step using rng primitives
#
# Translational invariance is ALWAYS checked and reported (it is a property of
# the algorithm, used to speed the check via orbit reduction when it holds). It
# need not be declared. D4 / point-group symmetry is never used.
#
# Exit code 0 iff: detailed balance holds, the chain is ergodic, and the state
# count matches the combinatorial formula. (Translational invariance is reported
# but does not by itself fail the run — see quadratic_field, an absolute-field
# algorithm that is correct yet not translation invariant.)
# ============================================================================

include(joinpath(@__DIR__, "dbc.jl"))

const SEP  = "="^64
const SEP2 = "-"^64

function parse_args(argv)
    isempty(argv) &&
        (println("Usage: julia [-t auto] check.jl <algorithm.jl> [-maxdepth N] [-parallel]"); exit(1))
    algfile = argv[1]
    isfile(algfile) || (println("ERROR: file not found: ", algfile); exit(1))
    maxdepth = 30; parallel = false; i = 2
    while i <= length(argv)
        if argv[i] == "-maxdepth"; maxdepth = parse(Int, argv[i+1]); i += 2
        elseif argv[i] == "-parallel"; parallel = true; i += 1
        else; println("ERROR: unknown option ", argv[i]); exit(1)
        end
    end
    (algfile, maxdepth, parallel)
end

# Canonical seed state: first N row-major sites, sorted types. Used as the
# ergodicity-reachability start.
function canonical_seed(types::Vector{Int}, n::Int)::CState
    st = sort(types); pos = [(r,c) for r in 1:n for c in 1:n]
    sort(NTuple{3,Int}[(pos[k][1], pos[k][2], st[k]) for k in 1:length(st)])
end

function run_checker(algfile, maxdepth, parallel, n, types, algo, energy)
    println(SEP); println("  DB_Julia  —  ", basename(algfile)); println(SEP)
    println("  nGrid      : ", n)
    println("  particles  : ", types)
    nthreads = Threads.nthreads()
    if parallel && nthreads == 1
        println("  parallel   : requested, but Julia has 1 thread — run as ",
                "`julia -t auto check.jl ...`. Falling back to serial.")
        parallel = false
    elseif parallel
        println("  parallel   : ON (", nthreads, " threads)")
    end
    seed = canonical_seed(types, n)
    println("  seed state : ", seed); println()

    println(SEP2); println("  Step 1: State enumeration"); println(SEP2)
    t1 = @elapsed states = enumerate_states(types, n)
    theo = theoretical_count(types, n); count_ok = length(states) == theo
    @printf("  States found : %d  (%.2fs)\n", length(states), t1)
    println("  State count  : ", count_ok ? "OK" : "MISMATCH",
            "  (found ", length(states), ", theoretical ", theo, ")")

    println(SEP2); println("  Step 2: tau-BFS (translational invariance + path enumeration)"); println(SEP2)
    moves = isdefined(Main, :MOVES) ? Vector{Tuple{Int,Int}}(Main.MOVES) : nothing
    local bfs
    try
        t2 = @elapsed (bfs = build_transitions(algo, energy, states, n, maxdepth; parallel=parallel, moves=moves))
        @printf("  BFS done  (%.2fs)\n", t2)
    catch e
        e isa CantHandle ? (println("  ERROR: ", e.msg); exit(1)) :
        e isa OverflowError ? (println("  ERROR: exact-arithmetic overflow (Int128) during BFS — ",
                                       "system too large for this build."); exit(1)) : rethrow(e)
    end
    println("  Translational : ", bfs.tau_free ? "PASS  — tau cancels in all leaf weights" :
            "FAIL  — " * bfs.tau_msg)
    bfs.tau_free || println("  (DB still checked directly from every state; orbit reduction not assumed.)")
    redparts = String[]
    bfs.tau_free && push!(redparts, "translation")
    bfs.species_free && push!(redparts, "species")
    isempty(bfs.pg_idx) || push!(redparts, "point group {" * join(pt_name.(bfs.pg_idx), ", ") * "}")
    @printf("  States BFS'd  : %d of %d  (%s)\n", bfs.nbfs, length(states),
            isempty(redparts) ? "all states (no equivariance)" :
            join(redparts, " + ") * " reduction (graph-certified)")

    println(SEP2); println("  Step 3: Ergodicity (reachability from seed)"); println(SEP2)
    t3 = @elapsed erg = check_ergodicity(bfs, seed)
    @printf("  Reachable : %d/%d  (%.2fs)\n", erg.reached, erg.total, t3)
    println("  Ergodicity : ", erg.ergodic ? "PASS" : "FAIL")

    println(SEP2); println("  Step 4: Detailed balance (exact-LP chambers + exact rational check)"); println(SEP2)
    local pass, viol, nch, m
    try
        tm = @elapsed m = build_dbmodel(bfs, energy)
        td = @elapsed ((pass, viol, nch) = run_db_check(m; parallel=parallel))
        @printf("  Chambers : %d   (model %.2fs, check %.2fs)\n", nch, tm, td)
        @printf("  DB pairs : %d of %d checked  (graph symmetry: %s)\n",
                length(m.check_pairs), length(m.pairs),
                isempty(m.sym_names) ? "none verified" : join(m.sym_names, ", "))
    catch e
        e isa CantHandle ? (println("  ERROR: ", e.msg); exit(1)) :
        e isa OverflowError ? (println("  ERROR: exact-arithmetic overflow (Int128) during DB check — ",
                                       "system too large for this build."); exit(1)) : rethrow(e)
    end
    if pass
        println("  Detailed bal. : PASS  — satisfied for all ", length(states), " states")
    else
        println("  Detailed bal. : FAIL  — ", length(viol), " violating (pair, chamber) record(s):")
        for v in first(viol, min(5, length(viol)))
            println("    s=", states[v[1]], "  t=", states[v[2]], "  (chamber ", v[3], ")")
        end
    end

    println(SEP); println("  SUMMARY"); println(SEP)
    println("  Translational : ", bfs.tau_free ? "PASS" : "FAIL")
    println("  State count   : ", count_ok ? "OK" : "MISMATCH")
    println("  Ergodicity    : ", erg.ergodic ? "PASS" : "FAIL")
    println("  Detailed bal. : ", pass ? "PASS" : "FAIL")
    println(SEP)

    exit((pass && erg.ergodic && count_ok) ? 0 : 1)
end

const _ALGFILE, _MAXDEPTH, _PARALLEL = parse_args(ARGS)
include(abspath(_ALGFILE))     # top-level: makes user methods visible to the call site
run_checker(_ALGFILE, _MAXDEPTH, _PARALLEL, Main.NGRID, Main.PARTICLE_TYPES, Main.algorithm, Main.energy)
