# =============================================================================
#  Compact Mixed-Integer Linear Programming Formulations for the CVRP
# =============================================================================
#
#  Implements 5 compact formulations in JuMP.jl + HiGHS.jl:
#    1. :MTZ         — 2-Index Capacitated Miller–Tucker–Zemlin (O(n²) vars/cons)
#    2. :LiftedMTZ   — 2-Index Desrochers–Laporte Lifted MTZ + Kara et al. bounds
#    3. :SCF         — 2-Index Single-Commodity Flow (Gavish & Graves, 1978)
#    4. :ThreeIndex  — 3-Index Vehicle-Flow Formulation (Wolsey Slide 10)
#    5. :MCF         — 2-Index Multi-Commodity Flow (Wolsey Slides 7 & 26, O(n³))
# =============================================================================

using JuMP
using HiGHS
import MathOptInterface as MOI

"""
    arc_structures(inst::CVRPInstance)

Construct directed arc list `arcs = [(i, j) for i in 1:N for j in 1:N if i != j]`,
dictionary `arc_index[(i, j)] -> a_idx`, and adjacency index lists `out_arcs[v]` and `in_arcs[v]`.
"""
function arc_structures(inst::CVRPInstance)
    N = inst.n + 1
    arcs = [(i, j) for i in 1:N for j in 1:N if i != j]
    arc_index = Dict(a => k for (k, a) in enumerate(arcs))
    out_arcs = [findall(a -> a[1] == v, arcs) for v in 1:N]
    in_arcs = [findall(a -> a[2] == v, arcs) for v in 1:N]
    return arcs, arc_index, out_arcs, in_arcs
end

"""
    build_cvrp_model(inst::CVRPInstance, formulation::Symbol; binary::Bool = true)

Build a JuMP model using `HiGHS.Optimizer` for `inst` under the chosen `formulation`:
- `:MTZ`: Basic 2-index Capacitated MTZ
- `:LiftedMTZ`: Desrochers–Laporte Lifted 2-index Capacitated MTZ
- `:SCF`: 2-index Single-Commodity Flow (Gavish & Graves, 1978)
- `:ThreeIndex`: 3-index Vehicle-Flow with symmetry breaking (Wolsey Slide 10)
- `:MCF`: 2-index Multi-Commodity Flow (Wolsey Slides 7 & 26)

Returns `(model = m, x_branch = Vector{VariableRef}, x_agg = Vector{AffExpr/VariableRef}, arcs = arcs)`.
- `x_branch` are the 0–1 variables branched on by Bonobo.jl (or set binary in MIP).
- `x_agg` gives the aggregated arc usage `x_a ∈ [0, 1]` for every `a in eachindex(arcs)`
  so `extract_routes(inst, arcs, value.(x_agg))` works uniformly across all 5 models.
"""
function build_cvrp_model(
    inst::CVRPInstance,
    formulation::Symbol;
    binary::Bool = true,
)
    N = inst.n + 1
    clients = 2:N
    K = inst.K
    Q = Float64(inst.Q)
    q = Float64.(inst.q)
    arcs, arc_index, out_arcs, in_arcs = arc_structures(inst)
    A_len = length(arcs)

    m = Model(HiGHS.Optimizer)
    set_silent(m)

    if formulation in (:MTZ, :LiftedMTZ)
        @variable(m, 0.0 <= x[1:A_len] <= 1.0)
        binary && set_binary.(x)
        @variable(m, q[i] <= u[i in clients] <= Q)

        @objective(m, Min, sum(inst.C[arcs[a][1], arcs[a][2]] * x[a] for a in 1:A_len))

        # Depot degree constraints
        @constraint(m, sum(x[a] for a in out_arcs[1]) == K)
        @constraint(m, sum(x[a] for a in in_arcs[1]) == K)

        # Client degree constraints
        for v in clients
            @constraint(m, sum(x[a] for a in out_arcs[v]) == 1)
            @constraint(m, sum(x[a] for a in in_arcs[v]) == 1)
        end

        # Forbid incompatible client-to-client arcs where q[i] + q[j] > Q
        for (a, (i, j)) in enumerate(arcs)
            if i >= 2 && j >= 2 && q[i] + q[j] > Q
                fix(x[a], 0.0; force = true)
            end
        end

        if formulation == :MTZ
            for i in clients, j in clients
                i == j && continue
                a_ij = arc_index[(i, j)]
                @constraint(m, u[i] - u[j] + Q * x[a_ij] <= Q - q[j])
            end
        else # :LiftedMTZ (Desrochers & Laporte, 1991; Kara, Laporte & Bektas, 2004)
            for i in clients, j in clients
                i == j && continue
                a_ij = arc_index[(i, j)]
                a_ji = arc_index[(j, i)]
                lift_coeff = max(0.0, Q - q[i] - q[j])
                @constraint(
                    m,
                    u[i] - u[j] + Q * x[a_ij] + lift_coeff * x[a_ji] <= Q - q[j]
                )
            end
            # Tightened node-load bounds
            for i in clients
                a_0i = arc_index[(1, i)]
                # 1. If i is the first client on a route (x_0i = 1), then u_i = q_i
                @constraint(m, u[i] + (Q - q[i]) * x[a_0i] <= Q)
                # 2. If i is followed by client j (x_ij = 1), then u_i <= Q - q_j
                @constraint(
                    m,
                    u[i] + sum(q[j] * x[arc_index[(i, j)]] for j in clients if j != i && q[i] + q[j] <= Q) <= Q
                )
                # 3. If i is preceded by client j (x_ji = 1), then u_i >= q_i + q_j
                @constraint(
                    m,
                    u[i] - sum(q[j] * x[arc_index[(j, i)]] for j in clients if j != i && q[i] + q[j] <= Q) >= q[i]
                )
            end
        end

        return (model = m, x_branch = x, x_agg = x, arcs = arcs)

    elseif formulation == :SCF
        # 2-Index Single-Commodity Flow (Gavish & Graves, 1978)
        @variable(m, 0.0 <= x[1:A_len] <= 1.0)
        binary && set_binary.(x)
        @variable(m, f[1:A_len] >= 0.0)

        @objective(m, Min, sum(inst.C[arcs[a][1], arcs[a][2]] * x[a] for a in 1:A_len))

        # Depot degree
        @constraint(m, sum(x[a] for a in out_arcs[1]) == K)
        @constraint(m, sum(x[a] for a in in_arcs[1]) == K)

        # Client degree
        for v in clients
            @constraint(m, sum(x[a] for a in out_arcs[v]) == 1)
            @constraint(m, sum(x[a] for a in in_arcs[v]) == 1)
        end

        # Flow leaving depot equals total client demand
        total_demand = sum(q[v] for v in clients)
        @constraint(m, sum(f[a] for a in out_arcs[1]) - sum(f[a] for a in in_arcs[1]) == total_demand)

        # Flow conservation at every client v: inflow - outflow = q[v]
        for v in clients
            @constraint(m, sum(f[a] for a in in_arcs[v]) - sum(f[a] for a in out_arcs[v]) == q[v])
        end

        # Tightened capacity-flow linking bounds on each arc a = (i, j)
        for (a, (i, j)) in enumerate(arcs)
            if j == 1
                # Vehicles return empty to the depot
                fix(f[a], 0.0; force = true)
            else
                if i >= 2 && q[i] + q[j] > Q
                    fix(x[a], 0.0; force = true)
                    fix(f[a], 0.0; force = true)
                else
                    @constraint(m, f[a] >= q[j] * x[a])
                    @constraint(m, f[a] <= (Q - q[i]) * x[a])
                end
            end
        end

        return (model = m, x_branch = x, x_agg = x, arcs = arcs)

    elseif formulation == :ThreeIndex
        # 3-Index Vehicle-Flow Formulation (Wolsey Slide 10)
        @variable(m, 0.0 <= x[1:A_len, 1:K] <= 1.0)
        @variable(m, 0.0 <= z[1:N, 1:K] <= 1.0)
        if binary
            set_binary.(x)
            set_binary.(z)
        end
        @variable(m, q[i] <= u[i in clients] <= Q)

        @objective(
            m,
            Min,
            sum(inst.C[arcs[a][1], arcs[a][2]] * x[a, k] for a in 1:A_len, k in 1:K)
        )

        # All K vehicles are used at the depot
        for k in 1:K
            fix(z[1, k], 1.0; force = true)
        end

        # Each client is visited by exactly one vehicle
        for v in clients
            @constraint(m, sum(z[v, k] for k in 1:K) == 1)
        end

        # Degree / flow conservation per vehicle k and node v
        for k in 1:K, v in 1:N
            @constraint(m, sum(x[a, k] for a in out_arcs[v]) == z[v, k])
            @constraint(m, sum(x[a, k] for a in in_arcs[v]) == z[v, k])
        end

        # Vehicle capacity constraint per vehicle k
        for k in 1:K
            @constraint(m, sum(q[v] * z[v, k] for v in clients) <= Q * z[1, k])
        end

        # Symmetry-breaking across identical vehicles: order vehicles by non-increasing load
        for k in 1:(K - 1)
            @constraint(
                m,
                sum(q[v] * z[v, k] for v in clients) >= sum(q[v] * z[v, k + 1] for v in clients)
            )
        end

        # Aggregated arc usage and lifted MTZ subtour elimination
        @expression(m, x_agg[a in 1:A_len], sum(x[a, k] for k in 1:K))
        for i in clients, j in clients
            i == j && continue
            a_ij = arc_index[(i, j)]
            a_ji = arc_index[(j, i)]
            lift_coeff = max(0.0, Q - q[i] - q[j])
            @constraint(
                m,
                u[i] - u[j] + Q * x_agg[a_ij] + lift_coeff * x_agg[a_ji] <= Q - q[j]
            )
        end

        x_branch = vcat(vec(x), [z[v, k] for v in clients for k in 1:K])
        return (model = m, x_branch = x_branch, x_agg = x_agg, arcs = arcs)

    elseif formulation == :MCF
        # 2-Index Multi-Commodity Flow Formulation (Wolsey Slides 7 & 26)
        @variable(m, 0.0 <= x[1:A_len] <= 1.0)
        binary && set_binary.(x)
        # f[a, w] >= 0: flow of commodity destined for client w ∈ clients on arc a
        @variable(m, f[1:A_len, w in clients] >= 0.0)

        @objective(m, Min, sum(inst.C[arcs[a][1], arcs[a][2]] * x[a] for a in 1:A_len))

        # Depot degree
        @constraint(m, sum(x[a] for a in out_arcs[1]) == K)
        @constraint(m, sum(x[a] for a in in_arcs[1]) == K)

        # Client degree
        for v in clients
            @constraint(m, sum(x[a] for a in out_arcs[v]) == 1)
            @constraint(m, sum(x[a] for a in in_arcs[v]) == 1)
        end

        # Multi-commodity flow conservation for each destination client w
        for w in clients
            # Origin (depot 1) sends q[w] units of commodity w
            @constraint(m, sum(f[a, w] for a in out_arcs[1]) - sum(f[a, w] for a in in_arcs[1]) == q[w])
            # Destination w absorbs q[w] units of commodity w
            @constraint(m, sum(f[a, w] for a in in_arcs[w]) - sum(f[a, w] for a in out_arcs[w]) == q[w])
            # Intermediate nodes v != w, 1 conserve flow of commodity w
            for v in clients
                v == w && continue
                @constraint(m, sum(f[a, w] for a in in_arcs[v]) - sum(f[a, w] for a in out_arcs[v]) == 0.0)
            end
        end

        # Arc-commodity and total arc-capacity linking constraints
        for (a, (i, j)) in enumerate(arcs)
            if j == 1
                # No commodity returns to the depot
                for w in clients
                    fix(f[a, w], 0.0; force = true)
                end
            else
                if i >= 2 && q[i] + q[j] > Q
                    fix(x[a], 0.0; force = true)
                    for w in clients
                        fix(f[a, w], 0.0; force = true)
                    end
                else
                    for w in clients
                        if i == w
                            # Commodity w is already delivered at node w and does not leave w
                            fix(f[a, w], 0.0; force = true)
                        elseif j == w
                            # If arc (i, w) is traversed, commodity w can be at most q[w]
                            @constraint(m, f[a, w] <= q[w] * x[a])
                        else
                            @constraint(m, f[a, w] <= q[w] * x[a])
                        end
                    end
                    # Aggregate load on arc (i, j)
                    @constraint(m, sum(f[a, w] for w in clients) >= q[j] * x[a])
                    @constraint(m, sum(f[a, w] for w in clients) <= (Q - q[i]) * x[a])
                end
            end
        end

        return (model = m, x_branch = x, x_agg = x, arcs = arcs)
    else
        error("Unknown CVRP formulation: $formulation")
    end
end
