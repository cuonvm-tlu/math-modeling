# =============================================================================
#  Custom LP-based Branch-and-Bound & Branch-and-Cut using Bonobo.jl
# =============================================================================
#
#  Directly mirrors the architecture of Lecture 5 (`tsp_mtz_compare.jl` and
#  `branch_and_cut_bonobo_pluto_v4.jl`):
#    1. `solve_cvrp_bonobo`: Plain LP-based Branch-and-Bound for the compact
#       formulations (`:MTZ`, `:LiftedMTZ`, `:SCF`, `:ThreeIndex`, `:MCF`).
#    2. `solve_cvrp_rcc_bnc`: 2-Index Branch-and-Cut separating Rounded Capacity
#       Cuts (RCCs, Wolsey Slide 11):
#           sum_{i ∉ S, j ∈ S} x_ij >= ceil(sum_{v ∈ S} d_v / Q)   ∀ S ⊆ V
# =============================================================================

using Bonobo
using JuMP
using HiGHS
import MathOptInterface as MOI

const BB = Bonobo

# -----------------------------------------------------------------------------
# Part 1: Plain LP-based Branch-and-Bound for Compact Formulations
# -----------------------------------------------------------------------------

mutable struct CVRPRoot
    model::JuMP.Model
    x::Vector{VariableRef}
    nb_nodes::Int
    node_limit::Int
end

mutable struct CVRPNode <: BB.AbstractNode
    std::BB.BnBNodeInfo
    lbs::Vector{Float64}
    ubs::Vector{Float64}
end

BB.get_branching_indices(root::CVRPRoot) = collect(1:length(root.x))

BB.get_relaxed_values(tree::BB.BnBTree{CVRPNode,CVRPRoot}, node::CVRPNode) =
    value.(tree.root.x)

function BB.get_branching_nodes_info(
    tree::BB.BnBTree{CVRPNode,CVRPRoot},
    node::CVRPNode,
    vidx::Int,
)
    lbs0, ubs0 = copy(node.lbs), copy(node.ubs)
    ubs0[vidx] = 0.0
    lbs1, ubs1 = copy(node.lbs), copy(node.ubs)
    lbs1[vidx] = 1.0
    return [(lbs = lbs0, ubs = ubs0), (lbs = lbs1, ubs = ubs1)]
end

function BB.evaluate_node!(tree::BB.BnBTree{CVRPNode,CVRPRoot}, node::CVRPNode)
    R = tree.root
    R.nb_nodes += 1
    for i in eachindex(R.x)
        if !is_fixed(R.x[i])
            set_lower_bound(R.x[i], node.lbs[i])
            set_upper_bound(R.x[i], node.ubs[i])
        end
    end
    optimize!(R.model)
    termination_status(R.model) == MOI.OPTIMAL || return NaN, NaN
    obj = objective_value(R.model)
    xvals = value.(R.x)
    if all(BB.is_approx_feasible(tree, v) for v in xvals)
        return obj, obj
    end
    return obj, NaN
end

function BB.terminated(tree::BB.BnBTree{CVRPNode,CVRPRoot})
    tree.root.nb_nodes >= tree.root.node_limit && return true
    return invoke(BB.terminated, Tuple{BB.BnBTree}, tree)
end

function bonobo_final_bound(tree)
    isempty(tree.nodes) && return tree.incumbent
    return min(tree.incumbent, minimum(nd.lb for nd in values(tree.nodes)))
end

"""
    solve_cvrp_bonobo(inst::CVRPInstance, formulation::Symbol; node_limit::Int = 10_000)

Solve `inst` with `formulation` using plain LP-based Branch-and-Bound in `Bonobo.jl`.
Returns `(obj, bound, gap, nodes, cpu_time, wall_time, status)`.
"""
function solve_cvrp_bonobo(
    inst::CVRPInstance,
    formulation::Symbol;
    node_limit::Int = 10_000,
)
    built = build_cvrp_model(inst, formulation; binary = false)
    m = built.model
    x = built.x_branch

    lbs0 = [is_fixed(v) ? fix_value(v) : 0.0 for v in x]
    ubs0 = [is_fixed(v) ? fix_value(v) : 1.0 for v in x]

    root = CVRPRoot(m, x, 0, node_limit)
    tree = BB.initialize(;
        traverse_strategy = BB.BestFirstSearch(),
        branch_strategy = BB.MOST_INFEASIBLE(),
        Node = CVRPNode,
        root = root,
        sense = :Min,
    )
    BB.set_root!(tree, (lbs = lbs0, ubs = ubs0))

    cpu_start = process_cpu_seconds()
    wall_start = time_ns()
    BB.optimize!(tree)
    wall_time = Float64(time_ns() - wall_start) / 1e9
    cpu_time = max(0.0, process_cpu_seconds() - cpu_start)

    obj = isempty(tree.solutions) ? Inf : BB.get_objective_value(tree)
    bound = bonobo_final_bound(tree)
    rel_gap = isfinite(obj) && obj > 0 ? max(0.0, 1.0 - bound / obj) : 1.0
    status = (isfinite(obj) && rel_gap <= 1e-5) ? "Optimal" : "NodeLimit"

    return (
        obj = obj,
        bound = bound,
        gap = rel_gap,
        nodes = root.nb_nodes,
        cpu_time = cpu_time,
        wall_time = wall_time,
        status = status,
    )
end

# -----------------------------------------------------------------------------
# Part 2: 2-Index Rounded Capacity Cut (RCC) Branch-and-Cut in Bonobo.jl
#         (Wolsey Slide 11 + Lecture 5 `tsp_mtz_compare.jl`)
# -----------------------------------------------------------------------------

mutable struct RCCRoot
    inst::CVRPInstance
    model::JuMP.Model
    x::Vector{VariableRef}
    arcs::Vector{Tuple{Int,Int}}
    nb_nodes::Int
    nb_cuts::Int
    node_limit::Int
    added_sets::Set{Vector{Int}}
end

mutable struct RCCNode <: BB.AbstractNode
    std::BB.BnBNodeInfo
    lbs::Vector{Float64}
    ubs::Vector{Float64}
end

BB.get_branching_indices(root::RCCRoot) = collect(1:length(root.x))

BB.get_relaxed_values(tree::BB.BnBTree{RCCNode,RCCRoot}, node::RCCNode) =
    value.(tree.root.x)

function BB.get_branching_nodes_info(
    tree::BB.BnBTree{RCCNode,RCCRoot},
    node::RCCNode,
    vidx::Int,
)
    lbs0, ubs0 = copy(node.lbs), copy(node.ubs)
    ubs0[vidx] = 0.0
    lbs1, ubs1 = copy(node.lbs), copy(node.ubs)
    lbs1[vidx] = 1.0
    return [(lbs = lbs0, ubs = ubs0), (lbs = lbs1, ubs = ubs1)]
end

"""
    separate_rcc_subsets(inst::CVRPInstance, arcs::Vector{Tuple{Int,Int}}, xval::Vector{Float64}; tol = 1e-5)

Find client subsets `S ⊆ {2, ..., n+1}` that violate the Rounded Capacity Cut (Wolsey Slide 11):
    sum_{i ∉ S, j ∈ S} x_ij >= ceil(sum_{v ∈ S} q_v / Q)
Checks connected components of the client support graph (at thresholds 0.5, 0.2, and 1e-4)
and greedy high-weight client clusters.
"""
function separate_rcc_subsets(
    inst::CVRPInstance,
    arcs::Vector{Tuple{Int,Int}},
    xval::Vector{Float64};
    tol::Float64 = 1e-5,
)
    N = inst.n + 1
    Q = inst.Q
    candidate_sets = Vector{Int}[]

    # 1. Connected components among clients 2:N at multiple support thresholds
    for thresh in (0.5, 0.25, 1e-4)
        adj = [Int[] for _ in 1:N]
        for (a, (i, j)) in enumerate(arcs)
            if i >= 2 && j >= 2 && xval[a] > thresh
                push!(adj[i], j)
                push!(adj[j], i)
            end
        end
        visited = falses(N)
        for start in 2:N
            visited[start] && continue
            comp = Int[]
            queue = [start]
            visited[start] = true
            while !isempty(queue)
                u = popfirst!(queue)
                push!(comp, u)
                for v in adj[u]
                    if !visited[v]
                        visited[v] = true
                        push!(queue, v)
                    end
                end
            end
            sort!(comp)
            push!(candidate_sets, comp)
        end
    end

    # 2. Greedy growth from each client seed to detect fractional capacity violations
    for seed_client in 2:N
        S = [seed_client]
        in_S = falses(N)
        in_S[seed_client] = true
        for _ in 1:(inst.n - 1)
            best_v = 0
            best_conn = 1e-4
            for v in 2:N
                in_S[v] && continue
                conn = 0.0
                for (a, (i, j)) in enumerate(arcs)
                    if (in_S[i] && j == v) || (i == v && in_S[j])
                        conn += xval[a]
                    end
                end
                if conn > best_conn
                    best_conn = conn
                    best_v = v
                end
            end
            best_v == 0 && break
            push!(S, best_v)
            in_S[best_v] = true
            push!(candidate_sets, sort(copy(S)))
        end
    end

    violated = Vector{Int}[]
    seen = Set{Vector{Int}}()
    for S in candidate_sets
        isempty(S) && continue
        S in seen && continue
        push!(seen, S)

        in_S = falses(N)
        demand_S = 0
        for v in S
            in_S[v] = true
            demand_S += inst.q[v]
        end
        rhs = ceil(Int, demand_S / Q)

        lhs = 0.0
        for (a, (i, j)) in enumerate(arcs)
            if !in_S[i] && in_S[j]
                lhs += xval[a]
            end
        end

        if lhs < rhs - tol
            push!(violated, S)
        end
    end

    return violated
end

function BB.evaluate_node!(tree::BB.BnBTree{RCCNode,RCCRoot}, node::RCCNode)
    R = tree.root
    R.nb_nodes += 1
    for i in eachindex(R.x)
        if !is_fixed(R.x[i])
            set_lower_bound(R.x[i], node.lbs[i])
            set_upper_bound(R.x[i], node.ubs[i])
        end
    end

    obj = NaN
    xvals = Float64[]
    for round in 1:8
        optimize!(R.model)
        termination_status(R.model) == MOI.OPTIMAL || return NaN, NaN
        obj = objective_value(R.model)
        xvals = value.(R.x)

        violated_sets = separate_rcc_subsets(R.inst, R.arcs, xvals)
        new_added = 0
        for S in violated_sets
            S in R.added_sets && continue
            push!(R.added_sets, S)
            in_S = Set(S)
            rhs = ceil(Int, sum(R.inst.q[v] for v in S) / R.inst.Q)
            @constraint(
                R.model,
                sum(R.x[a] for (a, (i, j)) in enumerate(R.arcs) if !(i in in_S) && (j in in_S)) >= rhs
            )
            R.nb_cuts += 1
            new_added += 1
        end
        new_added == 0 && break
    end

    if all(BB.is_approx_feasible(tree, v) for v in xvals)
        # Double-check no lazy RCC is violated at the integer point
        if isempty(separate_rcc_subsets(R.inst, R.arcs, xvals))
            return obj, obj
        end
    end
    return obj, NaN
end

function BB.terminated(tree::BB.BnBTree{RCCNode,RCCRoot})
    tree.root.nb_nodes >= tree.root.node_limit && return true
    return invoke(BB.terminated, Tuple{BB.BnBTree}, tree)
end

"""
    solve_cvrp_rcc_bnc(inst::CVRPInstance; node_limit::Int = 3_000)

Solve the 2-Index CVRP formulation with dynamic Rounded Capacity Cut (RCC)
separation in `Bonobo.jl` (Wolsey Slide 11 + Lecture 5 Branch-and-Cut).
Returns `(obj, bound, gap, nodes, cuts, cpu_time, wall_time, status)`.
"""
function solve_cvrp_rcc_bnc(inst::CVRPInstance; node_limit::Int = 3_000)
    N = inst.n + 1
    clients = 2:N
    K = inst.K
    Q = Float64(inst.Q)
    q = Float64.(inst.q)
    arcs, _, out_arcs, in_arcs = arc_structures(inst)
    A_len = length(arcs)

    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, 0.0 <= x[1:A_len] <= 1.0)
    @objective(m, Min, sum(inst.C[arcs[a][1], arcs[a][2]] * x[a] for a in 1:A_len))

    @constraint(m, sum(x[a] for a in out_arcs[1]) == K)
    @constraint(m, sum(x[a] for a in in_arcs[1]) == K)
    for v in clients
        @constraint(m, sum(x[a] for a in out_arcs[v]) == 1)
        @constraint(m, sum(x[a] for a in in_arcs[v]) == 1)
    end
    for (a, (i, j)) in enumerate(arcs)
        if i >= 2 && j >= 2 && q[i] + q[j] > Q
            fix(x[a], 0.0; force = true)
        end
    end

    lbs0 = [is_fixed(v) ? fix_value(v) : 0.0 for v in x]
    ubs0 = [is_fixed(v) ? fix_value(v) : 1.0 for v in x]

    root = RCCRoot(inst, m, x, arcs, 0, 0, node_limit, Set{Vector{Int}}())
    tree = BB.initialize(;
        traverse_strategy = BB.BestFirstSearch(),
        branch_strategy = BB.MOST_INFEASIBLE(),
        Node = RCCNode,
        root = root,
        sense = :Min,
    )
    BB.set_root!(tree, (lbs = lbs0, ubs = ubs0))

    cpu_start = process_cpu_seconds()
    wall_start = time_ns()
    BB.optimize!(tree)
    wall_time = Float64(time_ns() - wall_start) / 1e9
    cpu_time = max(0.0, process_cpu_seconds() - cpu_start)

    obj = isempty(tree.solutions) ? Inf : BB.get_objective_value(tree)
    bound = bonobo_final_bound(tree)
    rel_gap = isfinite(obj) && obj > 0 ? max(0.0, 1.0 - bound / obj) : 1.0
    status = (isfinite(obj) && rel_gap <= 1e-5) ? "Optimal" : "NodeLimit"

    return (
        obj = obj,
        bound = bound,
        gap = rel_gap,
        nodes = root.nb_nodes,
        cuts = root.nb_cuts,
        cpu_time = cpu_time,
        wall_time = wall_time,
        status = status,
    )
end
