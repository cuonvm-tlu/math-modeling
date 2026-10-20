# =============================================================================
#  Custom LP-based Branch-and-Bound for Compact CVRP Models using Bonobo.jl
# =============================================================================
#
#  Directly mirrors the architecture of Lecture 5 `tsp_mtz_compare.jl`:
#    - Builds the LP relaxation once (`binary = false`).
#    - Branches only on the binary decision variables `x_branch` while keeping
#      continuous variables (`u_i` or `f_ij`) in the LP relaxation.
#    - Uses `BestFirstSearch` and `MOST_INFEASIBLE` branching to measure the
#      pure impact of LP relaxation tightness on B&B tree size (`nb_nodes`).
# =============================================================================

using Bonobo
using JuMP
import MathOptInterface as MOI

const BB = Bonobo

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

    # Initialize bounds respecting variables fixed to 0 (e.g., incompatible arcs)
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
