# =============================================================================
#  CPU Timing, Root LP Evaluation, HiGHS MIP Solver Wrapper & CSV/LaTeX Export
# =============================================================================

using JuMP
using HiGHS
using DataFrames
using Statistics
using Printf
import MathOptInterface as MOI

"""
    process_cpu_seconds() -> Float64

Return total CPU time in seconds consumed by the Julia process (identical to
Lecture 5 `branch_and_cut_bonobo_pluto_v4.jl`).
"""
function process_cpu_seconds()
    if Sys.iswindows()
        creation = Ref{UInt64}(0)
        exit_time = Ref{UInt64}(0)
        kernel = Ref{UInt64}(0)
        user = Ref{UInt64}(0)
        handle = ccall((:GetCurrentProcess, "kernel32"), stdcall, Ptr{Cvoid}, ())
        ok = ccall(
            (:GetProcessTimes, "kernel32"), stdcall, Cint,
            (Ptr{Cvoid}, Ref{UInt64}, Ref{UInt64}, Ref{UInt64}, Ref{UInt64}),
            handle, creation, exit_time, kernel, user,
        )
        ok == 0 && error("GetProcessTimes failed")
        return (Float64(kernel[]) + Float64(user[])) * 1e-7
    else
        ticks = ccall(:clock, Clong, ())
        ticks == -1 && error("CPU clock unavailable")
        return Float64(ticks) / 1_000_000
    end
end

"""
    solve_root_lp(inst::CVRPInstance, formulation::Symbol)

Solve the linear programming relaxation (`binary = false`) of `formulation` on `inst`.
Returns `(bound = Float64, cpu_time = Float64, wall_time = Float64, num_vars = Int, num_cons = Int)`.
"""
function solve_root_lp(inst::CVRPInstance, formulation::Symbol)
    built = build_cvrp_model(inst, formulation; binary = false)
    m = built.model
    n_vars = num_variables(m)
    n_cons = num_constraints(m; count_variable_in_set_constraints = false)

    cpu_start = process_cpu_seconds()
    wall_start = time_ns()
    optimize!(m)
    wall_time = Float64(time_ns() - wall_start) / 1e9
    cpu_time = max(0.0, process_cpu_seconds() - cpu_start)

    bound = termination_status(m) == MOI.OPTIMAL ? objective_value(m) : NaN
    return (
        bound = bound,
        cpu_time = cpu_time,
        wall_time = wall_time,
        num_vars = n_vars,
        num_cons = n_cons,
    )
end

"""
    solve_cvrp_highs(inst::CVRPInstance, formulation::Symbol; time_limit::Float64 = 60.0)

Solve the full MILP model (`binary = true`) of `formulation` on `inst` using `HiGHS.Optimizer`.
Measures the execution time of the `optimize!(m)` call only (per `project/readme`).
Returns a NamedTuple with `(obj, bound, gap, nodes, cpu_time, wall_time, status, routes, valid, num_vars, num_cons)`.
"""
function solve_cvrp_highs(
    inst::CVRPInstance,
    formulation::Symbol;
    time_limit::Float64 = 60.0,
)
    built = build_cvrp_model(inst, formulation; binary = true)
    m = built.model
    set_time_limit_sec(m, time_limit)

    n_vars = num_variables(m)
    n_cons = num_constraints(m; count_variable_in_set_constraints = false)

    cpu_start = process_cpu_seconds()
    wall_start = time_ns()
    optimize!(m)
    wall_time = Float64(time_ns() - wall_start) / 1e9
    cpu_time = max(0.0, process_cpu_seconds() - cpu_start)

    term = termination_status(m)
    has_sol = has_values(m)
    obj = has_sol ? objective_value(m) : Inf
    bound = try
        objective_bound(m)
    catch
        NaN
    end

    # Gap formula from `project/readme`: gap = 1.0 - (Best known lower bound) / (value best known solution)
    rel_gap = if isfinite(obj) && isfinite(bound) && obj > 0
        max(0.0, 1.0 - bound / obj)
    else
        1.0
    end

    nodes = try
        Int(MOI.get(m, MOI.NodeCount()))
    catch
        -1
    end

    status_str = if term == MOI.OPTIMAL || (has_sol && rel_gap <= 1e-4)
        "Optimal"
    elseif term == MOI.INFEASIBLE
        "Infeasible"
    else
        "Not Optimal"
    end

    routes = Vector{Int}[]
    valid = false
    if has_sol
        x_vals = value.(built.x_agg)
        routes = extract_routes(inst, built.arcs, x_vals)
        val_res = validate_solution(inst, routes)
        valid = val_res.valid
    end

    return (
        obj = obj,
        bound = bound,
        gap = rel_gap,
        nodes = nodes,
        cpu_time = cpu_time,
        wall_time = wall_time,
        status = status_str,
        routes = routes,
        valid = valid,
        num_vars = n_vars,
        num_cons = n_cons,
    )
end

"""
    save_dataframe_csv(df::DataFrame, filepath::String)

Write a `DataFrame` to a standard comma-separated file without requiring external CSV packages.
"""
function save_dataframe_csv(df::DataFrame, filepath::String)
    mkpath(dirname(filepath))
    open(filepath, "w") do io
        cols = names(df)
        println(io, join(cols, ","))
        for row in eachrow(df)
            vals = [
                v isa AbstractFloat ? @sprintf("%.6f", v) : string(v)
                for v in values(row)
            ]
            println(io, join(vals, ","))
        end
    end
end
