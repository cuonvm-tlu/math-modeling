# =============================================================================
#  Capstone Numerical Experiments Suite (10 Runs per Configuration)
# =============================================================================
#
#  Follows the exact experimental protocol of:
#    - `project/readme` (averages over 10 runs: Instance info, Objective, CPU time,
#      Optimality Gap = 1 - LB/UB, Branch-and-Bound nodes, Status)
#    - `lecture5/tsp_mtz_compare.jl` (Root LP bounds, Bonobo B&B vs HiGHS MIP)
# =============================================================================

include(joinpath(@__DIR__, "..", "src", "instances.jl"))
include(joinpath(@__DIR__, "..", "src", "models.jl"))
include(joinpath(@__DIR__, "..", "src", "metrics.jl"))
include(joinpath(@__DIR__, "..", "src", "bonobo_bnb.jl"))

using DataFrames
using Statistics
using Printf

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

"""
Warm up Julia JIT compilation on a tiny instance for all formulations and solvers
before recording CPU times (matching `branch_and_cut_bonobo_pluto_v4.jl`).
"""
function warmup_all()
    print("Warming up Julia JIT compilation for all formulations... ")
    w = wolsey_instance()
    for form in (:MTZ, :LiftedMTZ, :SCF, :ThreeIndex, :MCF)
        solve_root_lp(w, form)
        solve_cvrp_highs(w, form; time_limit = 5.0)
        solve_cvrp_bonobo(w, form; node_limit = 100)
    end
    GC.gc()
    process_cpu_seconds()
    println("done.")
end

"""
Experiment A: Root LP Relaxation Bounds & Integrality Gaps (10 seeds per n).
"""
function run_experiment_root_bounds(; seeds = 1:10, n_values = (8, 10, 12, 15, 20, 25))
    println("\n" * "="^92)
    println("EXPERIMENT A: Root LP Relaxation Bounds & Integrality Gaps (Averaged over $(length(seeds)) runs)")
    println("="^92)

    raw_rows = NamedTuple[]
    forms = (:MTZ, :LiftedMTZ, :SCF, :ThreeIndex, :MCF)

    for n in n_values
        for seed in seeds
            inst = generate_cvrp_instance(n; seed = seed, topology = :uniform)
            # Obtain reference integer optimum (or best known UB) via SCF MIP
            ref_mip = solve_cvrp_highs(inst, :SCF; time_limit = 15.0)
            opt_val = ref_mip.obj

            for form in forms
                lp = solve_root_lp(inst, form)
                lp_gap_pct = 100.0 * max(0.0, (opt_val - lp.bound) / opt_val)
                push!(raw_rows, (
                    instance = inst.name,
                    n = n,
                    K = inst.K,
                    Q = inst.Q,
                    seed = seed,
                    formulation = String(form),
                    opt_ub = opt_val,
                    lp_bound = lp.bound,
                    lp_gap_pct = lp_gap_pct,
                    cpu_time = lp.cpu_time,
                    num_vars = lp.num_vars,
                    num_cons = lp.num_cons,
                ))
            end
        end
    end

    raw_df = DataFrame(raw_rows)
    save_dataframe_csv(raw_df, joinpath(RESULTS_DIR, "exp_root_bounds.csv"))

    # Aggregate over the 10 seeds per (n, formulation)
    sum_rows = NamedTuple[]
    @printf("%4s %5s | %-12s | %10s %10s %10s %10s %8s %8s\n",
            "n", "avg K", "Formulation", "Avg Opt", "Avg LP", "Avg Gap(%)", "Avg CPU(s)", "Vars", "Cons")
    println("-"^92)

    for n in n_values
        for form in forms
            sub = filter(r -> r.n == n && r.formulation == String(form), raw_df)
            row = (
                n = n,
                avg_K = mean(sub.K),
                formulation = String(form),
                avg_opt = mean(sub.opt_ub),
                avg_lp_bound = mean(sub.lp_bound),
                avg_lp_gap_pct = mean(sub.lp_gap_pct),
                std_lp_gap_pct = std(sub.lp_gap_pct),
                avg_cpu_time = mean(sub.cpu_time),
                num_vars = round(Int, mean(sub.num_vars)),
                num_cons = round(Int, mean(sub.num_cons)),
            )
            push!(sum_rows, row)
            @printf("%4d %5.1f | %-12s | %10.2f %10.2f %9.2f%% %10.4f %8d %8d\n",
                    row.n, row.avg_K, row.formulation, row.avg_opt, row.avg_lp_bound,
                    row.avg_lp_gap_pct, row.avg_cpu_time, row.num_vars, row.num_cons)
        end
        println("-"^92)
    end

    sum_df = DataFrame(sum_rows)
    save_dataframe_csv(sum_df, joinpath(RESULTS_DIR, "summary_root_bounds.csv"))
    return raw_df, sum_df
end

"""
Experiment B: Custom Plain LP-based Branch-and-Bound in Bonobo.jl (10 seeds per n).
Isolates the pure effect of LP relaxation tightness on B&B tree size without solver presolve/heuristics.
"""
function run_experiment_bonobo(; seeds = 1:10, n_values = (6, 8, 10, 12), node_limit = 3_000)
    println("\n" * "="^92)
    println("EXPERIMENT B: Plain LP Branch-and-Bound in Bonobo.jl (10 runs per n, node limit = $node_limit)")
    println("="^92)

    raw_rows = NamedTuple[]
    forms = (:MTZ, :LiftedMTZ, :SCF)

    for n in n_values
        for seed in seeds
            inst = generate_cvrp_instance(n; seed = seed, topology = :uniform)
            for form in forms
                res = solve_cvrp_bonobo(inst, form; node_limit = node_limit)
                push!(raw_rows, (
                    instance = inst.name,
                    n = n,
                    K = inst.K,
                    seed = seed,
                    formulation = String(form),
                    obj = res.obj,
                    bound = res.bound,
                    gap = res.gap,
                    nodes = res.nodes,
                    cpu_time = res.cpu_time,
                    status = res.status,
                ))
            end
        end
    end

    raw_df = DataFrame(raw_rows)
    save_dataframe_csv(raw_df, joinpath(RESULTS_DIR, "exp_bonobo_bnb.csv"))

    sum_rows = NamedTuple[]
    @printf("%4s %-12s | %10s %10s %10s %10s %10s %8s\n",
            "n", "Formulation", "Avg Obj", "Avg Bound", "Avg Gap", "Avg Nodes", "Avg CPU(s)", "#Opt/10")
    println("-"^92)

    for n in n_values
        for form in forms
            sub = filter(r -> r.n == n && r.formulation == String(form), raw_df)
            finite_objs = filter(isfinite, sub.obj)
            avg_obj = isempty(finite_objs) ? NaN : mean(finite_objs)
            n_opt = count(==("Optimal"), sub.status)
            row = (
                n = n,
                formulation = String(form),
                avg_obj = avg_obj,
                avg_bound = mean(sub.bound),
                avg_gap = mean(sub.gap),
                avg_nodes = mean(sub.nodes),
                avg_cpu_time = mean(sub.cpu_time),
                opt_count = n_opt,
            )
            push!(sum_rows, row)
            @printf("%4d %-12s | %10.2f %10.2f %10.4f %10.1f %10.4f %5d/10\n",
                    row.n, row.formulation, row.avg_obj, row.avg_bound,
                    row.avg_gap, row.avg_nodes, row.avg_cpu_time, row.opt_count)
        end
        println("-"^92)
    end

    sum_df = DataFrame(sum_rows)
    save_dataframe_csv(sum_df, joinpath(RESULTS_DIR, "summary_bonobo_bnb.csv"))
    return raw_df, sum_df
end

"""
Experiment C: Full MIP Solver Benchmark in HiGHS across 10 runs per size n.
Reports the exact metrics required by `project/readme`:
number of vertices, number of arcs, demands/vehicles, Optimal/Best objective,
CPU time (optimize! only), status, gap = 1.0 - LB/UB, and B&B nodes.
"""
function run_experiment_highs_mip(;
    seeds = 1:10,
    n_values = (10, 15, 20, 25, 30, 40),
    time_limit = 10.0,
)
    println("\n" * "="^98)
    println("EXPERIMENT C: HiGHS MIP Benchmark Across 10 Runs per Size n (Time Limit = $(time_limit)s)")
    println("="^98)

    raw_rows = NamedTuple[]

    for n in n_values
        # Include MCF up to n = 20 (since O(n³) has ~64,000 vars at n = 40)
        forms = n <= 20 ? (:MTZ, :LiftedMTZ, :SCF, :ThreeIndex, :MCF) : (:MTZ, :LiftedMTZ, :SCF, :ThreeIndex)
        for seed in seeds
            inst = generate_cvrp_instance(n; seed = seed, topology = :uniform)
            n_vertices = inst.n + 1
            n_arcs = n_vertices * (n_vertices - 1)

            for form in forms
                res = solve_cvrp_highs(inst, form; time_limit = time_limit)
                push!(raw_rows, (
                    instance = inst.name,
                    n = n,
                    vertices = n_vertices,
                    arcs = n_arcs,
                    demands = n,
                    K = inst.K,
                    Q = inst.Q,
                    seed = seed,
                    formulation = String(form),
                    num_vars = res.num_vars,
                    num_cons = res.num_cons,
                    obj = res.obj,
                    bound = res.bound,
                    gap = res.gap,
                    cpu_time = res.cpu_time,
                    nodes = res.nodes,
                    status = res.status,
                    valid = res.valid,
                ))
            end
        end
    end

    raw_df = DataFrame(raw_rows)
    save_dataframe_csv(raw_df, joinpath(RESULTS_DIR, "exp_highs_mip.csv"))

    sum_rows = NamedTuple[]
    @printf("%4s %5s %6s | %-11s | %9s %9s %9s %9s %9s %7s\n",
            "|V|", "|A|", "Avg K", "Formulation", "Avg Obj", "Avg LB", "Avg Gap", "CPU(s)", "Avg Nodes", "#Opt/10")
    println("-"^98)

    for n in n_values
        forms = n <= 20 ? (:MTZ, :LiftedMTZ, :SCF, :ThreeIndex, :MCF) : (:MTZ, :LiftedMTZ, :SCF, :ThreeIndex)
        for form in forms
            sub = filter(r -> r.n == n && r.formulation == String(form), raw_df)
            finite_objs = filter(isfinite, sub.obj)
            avg_obj = isempty(finite_objs) ? NaN : mean(finite_objs)
            n_opt = count(==("Optimal"), sub.status)
            row = (
                n = n,
                vertices = n + 1,
                arcs = (n + 1) * n,
                avg_K = mean(sub.K),
                formulation = String(form),
                num_vars = round(Int, mean(sub.num_vars)),
                num_cons = round(Int, mean(sub.num_cons)),
                avg_obj = avg_obj,
                avg_bound = mean(sub.bound),
                avg_gap = mean(sub.gap),
                avg_gap_pct = 100.0 * mean(sub.gap),
                avg_cpu_time = mean(sub.cpu_time),
                avg_nodes = mean(sub.nodes),
                opt_count = n_opt,
            )
            push!(sum_rows, row)
            @printf("%4d %5d %6.1f | %-11s | %9.2f %9.2f %9.4f %9.3f %9.1f %4d/10\n",
                    row.vertices, row.arcs, row.avg_K, row.formulation,
                    row.avg_obj, row.avg_bound, row.avg_gap,
                    row.avg_cpu_time, row.avg_nodes, row.opt_count)
        end
        println("-"^98)
    end

    sum_df = DataFrame(sum_rows)
    save_dataframe_csv(sum_df, joinpath(RESULTS_DIR, "summary_highs_mip.csv"))
    return raw_df, sum_df
end

"""
Experiment D: Sensitivity Analysis at n = 18 across Topology (:uniform vs :clustered)
and Capacity Tightness alpha in (0.70, 0.85, 0.95) over 10 seeds.
"""
function run_experiment_sensitivity(; seeds = 1:10, n = 18, time_limit = 8.0)
    println("\n" * "="^92)
    println("EXPERIMENT D: Sensitivity to Capacity Tightness & Topology (n = $n, 10 runs per setting)")
    println("="^92)

    raw_rows = NamedTuple[]
    configs = (
        (topology = :uniform,   alpha = 0.70, label = "U-Center (alpha=0.70, Loose)"),
        (topology = :uniform,   alpha = 0.85, label = "U-Center (alpha=0.85, Medium)"),
        (topology = :uniform,   alpha = 0.95, label = "U-Center (alpha=0.95, Tight)"),
        (topology = :clustered, alpha = 0.85, label = "C-Corner (alpha=0.85, Clustered)"),
    )
    forms = (:MTZ, :LiftedMTZ, :SCF)

    for cfg in configs
        for seed in seeds
            inst = generate_cvrp_instance(n; seed = seed, topology = cfg.topology, alpha = cfg.alpha)
            for form in forms
                lp = solve_root_lp(inst, form)
                mip = solve_cvrp_highs(inst, form; time_limit = time_limit)
                push!(raw_rows, (
                    setting = cfg.label,
                    topology = String(cfg.topology),
                    alpha = cfg.alpha,
                    n = n,
                    K = inst.K,
                    seed = seed,
                    formulation = String(form),
                    lp_bound = lp.bound,
                    obj = mip.obj,
                    bound = mip.bound,
                    gap = mip.gap,
                    cpu_time = mip.cpu_time,
                    nodes = mip.nodes,
                    status = mip.status,
                ))
            end
        end
    end

    raw_df = DataFrame(raw_rows)
    save_dataframe_csv(raw_df, joinpath(RESULTS_DIR, "exp_sensitivity.csv"))

    sum_rows = NamedTuple[]
    @printf("%-32s | %5s %-10s | %8s %8s %8s %8s %9s %7s\n",
            "Setting", "Avg K", "Model", "LP Bound", "Avg Obj", "Avg Gap", "CPU(s)", "Avg Nodes", "#Opt/10")
    println("-"^104)

    for cfg in configs
        for form in forms
            sub = filter(r -> r.setting == cfg.label && r.formulation == String(form), raw_df)
            row = (
                setting = cfg.label,
                topology = String(cfg.topology),
                alpha = cfg.alpha,
                avg_K = mean(sub.K),
                formulation = String(form),
                avg_lp_bound = mean(sub.lp_bound),
                avg_obj = mean(sub.obj),
                avg_gap = mean(sub.gap),
                avg_cpu_time = mean(sub.cpu_time),
                avg_nodes = mean(sub.nodes),
                opt_count = count(==("Optimal"), sub.status),
            )
            push!(sum_rows, row)
            @printf("%-32s | %5.1f %-10s | %8.2f %8.2f %8.4f %8.3f %9.1f %4d/10\n",
                    row.setting, row.avg_K, row.formulation, row.avg_lp_bound,
                    row.avg_obj, row.avg_gap, row.avg_cpu_time, row.avg_nodes, row.opt_count)
        end
        println("-"^104)
    end

    sum_df = DataFrame(sum_rows)
    save_dataframe_csv(sum_df, joinpath(RESULTS_DIR, "summary_sensitivity.csv"))
    return raw_df, sum_df
end

"""
Export auto-generated LaTeX booktabs tables to `results/summary_tables.tex`.
"""
function export_latex_tables(sum_lp::DataFrame, sum_bnb::DataFrame, sum_mip::DataFrame, sum_sens::DataFrame)
    tex_path = joinpath(RESULTS_DIR, "summary_tables.tex")
    mkpath(RESULTS_DIR)
    open(tex_path, "w") do io
        println(io, "% Auto-generated LaTeX tables from `scripts/run_experiments.jl` (10 runs per setting)")
        println(io, "\\begin{table}[htbp]")
        println(io, "\\centering")
        println(io, "\\caption{Root LP Relaxation Bounds and Integrality Gaps (\\%) Averaged over 10 Instances per Size \$n\$}")
        println(io, "\\label{tab:root_lp}")
        println(io, "\\small")
        println(io, "\\begin{tabular}{rrlrrrrrr}")
        println(io, "\\toprule")
        println(io, "\$n\$ & \$\\bar{K}\$ & Formulation & Vars & Cons & Avg \$z^*\$ & Avg \$LB_{\\text{LP}}\$ & Gap (\\%) & LP CPU (s) \\\\")
        println(io, "\\midrule")
        for r in eachrow(sum_lp)
            @printf(io, "%d & %.1f & \\texttt{%s} & %d & %d & %.2f & %.2f & %.2f \\pm %.2f & %.4f \\\\\n",
                    r.n, r.avg_K, r.formulation, r.num_vars, r.num_cons,
                    r.avg_opt, r.avg_lp_bound, r.avg_lp_gap_pct, r.std_lp_gap_pct, r.avg_cpu_time)
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}\n")

        println(io, "\\begin{table}[htbp]")
        println(io, "\\centering")
        println(io, "\\caption{Plain LP Branch-and-Bound in \\texttt{Bonobo.jl} (Averaged over 10 Runs, Node Limit = 3,000)}")
        println(io, "\\label{tab:bonobo_bnb}")
        println(io, "\\small")
        println(io, "\\begin{tabular}{rlrrrrrr}")
        println(io, "\\toprule")
        println(io, "\$n\$ & Formulation & Avg Obj & Avg Bound & Avg Gap & Avg B\\&B Nodes & Avg CPU (s) & \\#Opt/10 \\\\")
        println(io, "\\midrule")
        for r in eachrow(sum_bnb)
            @printf(io, "%d & \\texttt{%s} & %.2f & %.2f & %.4f & %.1f & %.4f & %d/10 \\\\\n",
                    r.n, r.formulation, r.avg_obj, r.avg_bound, r.avg_gap,
                    r.avg_nodes, r.avg_cpu_time, r.opt_count)
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}\n")

        println(io, "\\begin{table}[htbp]")
        println(io, "\\centering")
        println(io, "\\caption{Full MILP Performance in \\texttt{HiGHS} Averaged over 10 Runs per Size \$n\$ (Time Limit = 10s)}")
        println(io, "\\label{tab:highs_mip}")
        println(io, "\\small")
        println(io, "\\begin{tabular}{rrrlrrrrrrr}")
        println(io, "\\toprule")
        println(io, "\$|V_0|\$ & \$|A|\$ & \$\\bar{K}\$ & Formulation & Vars & Cons & Avg Obj & Avg LB & Avg Gap & CPU (s) & Avg Nodes & \\#Opt \\\\")
        println(io, "\\midrule")
        for r in eachrow(sum_mip)
            @printf(io, "%d & %d & %.1f & \\texttt{%s} & %d & %d & %.2f & %.2f & %.4f & %.3f & %.1f & %d/10 \\\\\n",
                    r.vertices, r.arcs, r.avg_K, r.formulation, r.num_vars, r.num_cons,
                    r.avg_obj, r.avg_bound, r.avg_gap, r.avg_cpu_time, r.avg_nodes, r.opt_count)
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}\n")
    end
    println("\nExported LaTeX tables to $tex_path")
end

function main()
    warmup_all()
    _, sum_lp   = run_experiment_root_bounds()
    _, sum_bnb  = run_experiment_bonobo()
    _, sum_mip  = run_experiment_highs_mip()
    _, sum_sens = run_experiment_sensitivity()
    export_latex_tables(sum_lp, sum_bnb, sum_mip, sum_sens)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
