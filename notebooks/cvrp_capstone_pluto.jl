### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook as a regular Julia script, the @bind macro is mocked to return the default value of the widget.
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 10000000-0000-0000-0000-000000000001
begin
    using Bonobo
    using JuMP
    using HiGHS
    using DataFrames
    using PlutoUI
    using Printf
    using Random
    using Statistics

    import MathOptInterface as MOI

    include(joinpath(@__DIR__, "..", "src", "instances.jl"))
    include(joinpath(@__DIR__, "..", "src", "models.jl"))
    include(joinpath(@__DIR__, "..", "src", "metrics.jl"))
    include(joinpath(@__DIR__, "..", "src", "bonobo_bnb.jl"))
end

# ╔═╡ 10000000-0000-0000-0000-000000000002
md"""
# Capacitated Vehicle Routing Problem (CVRP): Compact MILP Formulations & Branch-and-Bound Analysis

**Master of Data Science — Mathematical Modelling Capstone Project**

This interactive Pluto notebook accompanies our capstone report and implements **five compact Mixed-Integer Linear Programming (MILP) formulations** for the Single-Depot Capacitated Vehicle Routing Problem (CVRP), building on Lecture 5 (`tsp_mtz_compare.jl`, `branch_and_cut_bonobo_pluto_v4.jl`) and Lecture 6 (`Wolsey-VRP.pdf`):

1. **Basic 2-Index MTZ (`:MTZ`)**: Cumulative vehicle load variables $u_i \in [d_i, Q]$ with big-$M$ subtour elimination $u_i - u_j + Q x_{ij} \le Q - d_j$.
2. **Desrochers–Laporte Lifted 2-Index MTZ (`:LiftedMTZ`)**: Lifts the reverse arc variable $x_{ji}$ with coefficient $\max(0, Q - d_i - d_j)$ and tightens node-load bounds.
3. **2-Index Single-Commodity Flow (`:SCF`)** *(Gavish & Graves, 1978)*: Continuous arc-flow variables $f_{ij} \ge 0$ with tightened capacity linking $d_j x_{ij} \le f_{ij} \le (Q - d_i) x_{ij}$.
4. **3-Index Vehicle-Flow (`:ThreeIndex`)** *(Wolsey Slide 10)*: Explicit vehicle-indexed arc variables $x_{ij}^k \in \{0,1\}$ and assignment variables $z_i^k \in \{0,1\}$ with load-ordered symmetry-breaking constraints.
5. **2-Index Multi-Commodity Flow (`:MCF`)** *(Wolsey Slides 7 & 26)*: Disaggregated commodity flows $f_{ij}^w \ge 0$ from depot $0$ to each client $w \in V$.
"""

# ╔═╡ 10000000-0000-0000-0000-000000000003
TableOfContents()

# ╔═╡ 10000000-0000-0000-0000-000000000004
md"""
## 1. Benchmark Verification: Wolsey's Illustrative CVRP Instance

We first verify all 5 formulations on the 3-client illustrative instance from Laurence A. Wolsey's Lecture 6 slides (`Wolsey-VRP.pdf`, Slides 19–20), where $n = 3$, $d = (2, 4, 5)$, $Q = 10$, $K = 2$, and the known integer optimum is $z^* = 49$.
"""

# ╔═╡ 10000000-0000-0000-0000-000000000005
wolsey_comparison = let
    w_inst = wolsey_instance()
    forms = (:MTZ, :LiftedMTZ, :SCF, :ThreeIndex, :MCF)
    DataFrame([
        let
            lp = solve_root_lp(w_inst, f)
            mip = solve_cvrp_highs(w_inst, f; time_limit = 10.0)
            bnb = solve_cvrp_bonobo(w_inst, f; node_limit = 1_000)
            (
                formulation = String(f),
                num_vars = lp.num_vars,
                num_cons = lp.num_cons,
                root_lp_bound = round(lp.bound; digits = 2),
                lp_gap_pct = round(100.0 * (mip.obj - lp.bound) / mip.obj; digits = 2),
                mip_objective = round(mip.obj; digits = 2),
                bonobo_objective = round(bnb.obj; digits = 2),
                bonobo_nodes = bnb.nodes,
                highs_nodes = mip.nodes,
                routes_valid = mip.valid,
            )
        end
        for f in forms
    ])
end

# ╔═╡ 10000000-0000-0000-0000-000000000006
md"""
## 2. Interactive Instance Generator & Solver Explorer

Adjust the sliders below to generate a synthetic CVRP instance and inspect the optimal vehicle tours:

- **Number of clients ($n$)**: $(@bind n_clients Slider(6:2:20, default=10, show_value=true))
- **Random seed**: $(@bind inst_seed Slider(1:10, default=1, show_value=true))
- **Spatial topology**: $(@bind inst_topo Select([:uniform => "Uniform (Center Depot)", :clustered => "Clustered (Corner Depot)"]))
- **Capacity tightness ($\alpha$)**: $(@bind inst_alpha Slider(0.70:0.05:0.95, default=0.85, show_value=true))
- **Formulation to visualize**: $(@bind viz_form Select([:SCF => "Single-Commodity Flow (:SCF)", :LiftedMTZ => "Lifted MTZ (:LiftedMTZ)", :MTZ => "Basic MTZ (:MTZ)", :ThreeIndex => "3-Index Vehicle Flow (:ThreeIndex)", :MCF => "Multi-Commodity Flow (:MCF)"]))
"""

# ╔═╡ 10000000-0000-0000-0000-000000000007
current_instance = generate_cvrp_instance(
    n_clients;
    seed = inst_seed,
    topology = inst_topo,
    alpha = inst_alpha,
)

# ╔═╡ 10000000-0000-0000-0000-000000000008
viz_result = solve_cvrp_highs(current_instance, viz_form; time_limit = 10.0)

# ╔═╡ 10000000-0000-0000-0000-000000000009
"""
Render an SVG diagram of `inst` and `routes` directly in HTML (zero external plotting dependencies).
"""
function render_routes_svg(inst::CVRPInstance, routes::Vector{Vector{Int}}, obj::Float64)
    W, H = 560, 440
    pad = 45.0
    scale_x(x) = pad + (x / 100.0) * (W - 2 * pad)
    scale_y(y) = H - pad - (y / 100.0) * (H - 2 * pad)

    palette = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#9333ea", "#0891b2", "#be185d", "#4f46e5"]
    io = IOBuffer()
    println(io, """<svg width="$W" height="$H" viewBox="0 0 $W $H" style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; font-family:sans-serif;">""")
    println(io, """<text x="$(W/2)" y="24" text-anchor="middle" font-size="14" font-weight="bold" fill="#0f172a">Instance $(inst.name) — K=$(inst.K), Q=$(inst.Q), Cost=$(round(obj, digits=1))</text>""")

    # Draw routes
    for (r_idx, route) in enumerate(routes)
        col = palette[mod1(r_idx, length(palette))]
        for pos in 1:(length(route) - 1)
            u, v = route[pos], route[pos + 1]
            x1, y1 = scale_x(inst.coords[u, 1]), scale_y(inst.coords[u, 2])
            x2, y2 = scale_x(inst.coords[v, 1]), scale_y(inst.coords[v, 2])
            println(io, """<line x1="$x1" y1="$y1" x2="$x2" y2="$y2" stroke="$col" stroke-width="2.4" stroke-opacity="0.85"/>""")
        end
    end

    # Draw client nodes
    for i in 2:(inst.n + 1)
        cx, cy = scale_x(inst.coords[i, 1]), scale_y(inst.coords[i, 2])
        println(io, """<circle cx="$cx" cy="$cy" r="10" fill="#ffffff" stroke="#334155" stroke-width="2"/>""")
        println(io, """<text x="$cx" y="$(cy + 4)" text-anchor="middle" font-size="10" font-weight="bold" fill="#0f172a">$(i - 1)</text>""")
        println(io, """<text x="$(cx + 13)" y="$(cy - 6)" font-size="9" fill="#475569">d=$(inst.q[i])</text>""")
    end

    # Draw depot node (square)
    dx, dy = scale_x(inst.coords[1, 1]), scale_y(inst.coords[1, 2])
    println(io, """<rect x="$(dx - 11)" y="$(dy - 11)" width="22" height="22" rx="3" fill="#0f172a" stroke="#ffffff" stroke-width="2"/>""")
    println(io, """<text x="$dx" y="$(dy + 4)" text-anchor="middle" font-size="10" font-weight="bold" fill="#ffffff">D</text>""")

    println(io, "</svg>")
    return HTML(String(take!(io)))
end

# ╔═╡ 10000000-0000-0000-0000-000000000010
render_routes_svg(current_instance, viz_result.routes, viz_result.obj)

# ╔═╡ 10000000-0000-0000-0000-000000000011
md"""
## 3. Cross-Formulation Comparison on Current Instance

The table below compares all five compact formulations on `$(current_instance.name)`:
"""

# ╔═╡ 10000000-0000-0000-0000-000000000012
instance_comparison_table = let
    forms = (:MTZ, :LiftedMTZ, :SCF, :ThreeIndex, :MCF)
    DataFrame([
        let
            lp = solve_root_lp(current_instance, f)
            mip = solve_cvrp_highs(current_instance, f; time_limit = 8.0)
            (
                formulation = String(f),
                variables = lp.num_vars,
                constraints = lp.num_cons,
                lp_bound = round(lp.bound; digits = 2),
                mip_obj = round(mip.obj; digits = 2),
                mip_bound = round(mip.bound; digits = 2),
                mip_gap = round(mip.gap; digits = 4),
                highs_nodes = mip.nodes,
                cpu_time_s = round(mip.cpu_time; digits = 4),
                status = mip.status,
                valid = mip.valid,
            )
        end
        for f in forms
    ])
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Bonobo = "f7b14807-3d4d-461a-888a-05dd4bca8bc3"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
HiGHS = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Bonobo = "~0.1.5"
DataFrames = "~1.8.2"
HiGHS = "~1.24.1"
JuMP = "~1.31.1"
MathOptInterface = "~1.52.0"
PlutoUI = "~0.7.83"
"""

# ╔═╡ Cell order:
# ╟─10000000-0000-0000-0000-000000000002
# ╠═10000000-0000-0000-0000-000000000001
# ╠═10000000-0000-0000-0000-000000000003
# ╟─10000000-0000-0000-0000-000000000004
# ╠═10000000-0000-0000-0000-000000000005
# ╟─10000000-0000-0000-0000-000000000006
# ╠═10000000-0000-0000-0000-000000000007
# ╠═10000000-0000-0000-0000-000000000008
# ╠═10000000-0000-0000-0000-000000000009
# ╠═10000000-0000-0000-0000-000000000010
# ╟─10000000-0000-0000-0000-000000000011
# ╠═10000000-0000-0000-0000-000000000012
# ╟─00000000-0000-0000-0000-000000000001
