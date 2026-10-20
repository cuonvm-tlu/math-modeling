# =============================================================================
#  CVRP Instance Data Structure, Synthetic Instance Generator & Validator
# =============================================================================
#
#  Node Indexing Convention (matching Lecture 5 `tsp_mtz_compare.jl`):
#    - Node 1 is the central depot (denoted 0 in mathematical notation).
#    - Nodes 2:(n+1) are the n clients (denoted 1..n in mathematical notation).
#    - `q` has length n+1 with `q[1] = 0` and `q[i] > 0` for `i in 2:(n+1)`.
#    - `C` is the (n+1) × (n+1) symmetric/directed travel cost matrix.
# =============================================================================

using Random
using Printf

struct CVRPInstance
    name::String
    n::Int                      # number of clients (total nodes = n + 1)
    K::Int                      # number of identical vehicles
    Q::Int                      # vehicle capacity
    q::Vector{Int}              # node demands of length n + 1 (q[1] = 0 for depot)
    coords::Matrix{Float64}     # (n + 1) × 2 matrix of (x, y) coordinates
    C::Matrix{Float64}          # (n + 1) × (n + 1) travel cost matrix
    seed::Int                   # random seed used for generation
    topology::Symbol            # :uniform, :clustered, or :benchmark
    alpha::Float64              # capacity tightness parameter
end

"""
    wolsey_instance() -> CVRPInstance

Return the 3-client illustrative CVRP instance from Laurence A. Wolsey's
Lecture 6 slides (`Wolsey-VRP.pdf`, Slides 19–20):
- n = 3 clients, d = (2, 4, 5), vehicle capacity Q = 10, K = 2 vehicles.
- Costs: c_01 = 11, c_02 = 12, c_03 = 10, c_12 = 16, c_13 = 18, c_23 = 5.
- Optimal integer tour cost = 49.0:
    Route 1: 0 -> 1 -> 0 (load = 2, cost = 22)
    Route 2: 0 -> 2 -> 3 -> 0 (load = 9, cost = 27)
"""
function wolsey_instance()
    n = 3
    K = 2
    Q = 10
    q = [0, 2, 4, 5]
    coords = [
        0.0  0.0;   # depot (0)
        -2.0 8.0;   # client 1
        6.0  5.0;   # client 2
        6.0  0.0;   # client 3
    ]
    C = [
         0.0 11.0 12.0 10.0;
        11.0  0.0 16.0 18.0;
        12.0 16.0  0.0  5.0;
        10.0 18.0  5.0  0.0;
    ]
    return CVRPInstance("Wolsey-n3-k2", n, K, Q, q, coords, C, 0, :benchmark, 0.55)
end

"""
    ffd_bins_needed(demands::Vector{Int}, Q::Int) -> Int

Compute the number of bins of capacity `Q` required by First-Fit Decreasing (FFD)
to guarantee that a generated instance with `K` vehicles is strictly feasible.
"""
function ffd_bins_needed(demands::Vector{Int}, Q::Int)
    sorted_d = sort(demands; rev = true)
    bins = Int[]
    for d in sorted_d
        placed = false
        for b in eachindex(bins)
            if bins[b] + d <= Q
                bins[b] += d
                placed = true
                break
            end
        end
        if !placed
            push!(bins, d)
        end
    end
    return length(bins)
end

"""
    generate_cvrp_instance(n::Int;
                           seed::Int = 1,
                           Q::Int = 100,
                           alpha::Float64 = 0.85,
                           topology::Symbol = :uniform,
                           demand_range::Tuple{Int,Int} = (5, 25)) -> CVRPInstance

Deterministically generate a synthetic CVRP instance with `n` clients:
- `topology == :uniform` (`U-Center`):
    Depot at center `(50, 50)`, client coordinates drawn i.i.d. uniformly in `[0, 100]²`.
- `topology == :clustered` (`C-Corner`):
    Depot at corner `(0, 0)`, clients drawn around 3 random cluster centers in `[20, 90]²`
    with Gaussian standard deviation `σ = 10.0` (clamped to `[0, 100]²`).
- Travel costs `C[i, j]` use TSPLIB `EUC_2D` rounded Euclidean distances:
    `round(Int, hypot(x[i] - x[j], y[i] - y[j]))`, with triangle inequality enforced.
- Client demands `q[2:(n+1)]` are drawn uniformly from `demand_range[1]:demand_range[2]`.
- Fleet size `K = max(2, ceil(Int, sum(q) / (alpha * Q)), ffd_bins_needed(q[2:end], Q))`.
"""
function generate_cvrp_instance(
    n::Int;
    seed::Int = 1,
    Q::Int = 100,
    alpha::Float64 = 0.85,
    topology::Symbol = :uniform,
    demand_range::Tuple{Int,Int} = (5, 25),
)
    rng = MersenneTwister(seed)
    N = n + 1
    coords = zeros(Float64, N, 2)

    if topology == :uniform
        coords[1, 1] = 50.0
        coords[1, 2] = 50.0
        for i in 2:N
            coords[i, 1] = 100.0 * rand(rng)
            coords[i, 2] = 100.0 * rand(rng)
        end
    elseif topology == :clustered
        coords[1, 1] = 0.0
        coords[1, 2] = 0.0
        n_clusters = 3
        centers = 20.0 .+ 70.0 .* rand(rng, n_clusters, 2)
        for i in 2:N
            c_idx = mod1(i + rand(rng, 0:n_clusters-1), n_clusters)
            coords[i, 1] = clamp(centers[c_idx, 1] + 10.0 * randn(rng), 0.0, 100.0)
            coords[i, 2] = clamp(centers[c_idx, 2] + 10.0 * randn(rng), 0.0, 100.0)
        end
    else
        error("Unsupported topology: $topology (use :uniform or :clustered)")
    end

    q = zeros(Int, N)
    dmin, dmax = demand_range
    for i in 2:N
        q[i] = rand(rng, dmin:dmax)
    end

    total_demand = sum(q)
    k_target = ceil(Int, total_demand / (alpha * Q))
    k_ffd = ffd_bins_needed(q[2:N], Q)
    K = max(2, k_target, k_ffd)

    C = zeros(Float64, N, N)
    for i in 1:N, j in 1:N
        if i != j
            dist = hypot(coords[i, 1] - coords[j, 1], coords[i, 2] - coords[j, 2])
            C[i, j] = max(1.0, Float64(round(Int, dist)))
        end
    end
    # Floyd-Warshall pass to guarantee strict metric triangle inequality after rounding
    for k in 1:N, i in 1:N, j in 1:N
        if i != j && C[i, k] + C[k, j] < C[i, j]
            C[i, j] = C[i, k] + C[k, j]
        end
    end

    prefix = topology == :uniform ? "U" : "C"
    name = @sprintf("%s-n%02d-k%d-s%02d", prefix, n, K, seed)
    return CVRPInstance(name, n, K, Q, q, coords, C, seed, topology, alpha)
end

"""
    extract_routes(inst::CVRPInstance, arcs::Vector{Tuple{Int,Int}}, xval::AbstractVector{<:Real}; tol = 0.5)

Extract the `K` vehicle routes (each represented as `[1, v_1, ..., v_m, 1]`)
from a binary arc-value vector `xval`.
"""
function extract_routes(
    inst::CVRPInstance,
    arcs::Vector{Tuple{Int,Int}},
    xval::AbstractVector{<:Real};
    tol::Float64 = 0.5,
)
    N = inst.n + 1
    adj = [Int[] for _ in 1:N]
    for (a_idx, (i, j)) in enumerate(arcs)
        if xval[a_idx] > tol
            push!(adj[i], j)
        end
    end

    routes = Vector{Int}[]
    for start_client in adj[1]
        route = [1, start_client]
        curr = start_client
        steps = 0
        while curr != 1 && steps <= N + 2
            steps += 1
            isempty(adj[curr]) && break
            nxt = popfirst!(adj[curr])
            push!(route, nxt)
            curr = nxt
        end
        push!(routes, route)
    end
    return routes
end

"""
    validate_solution(inst::CVRPInstance, routes::Vector{Vector{Int}}; atol = 1e-5)

Verify that `routes` forms a valid CVRP solution for `inst`:
1. Exactly `inst.K` routes, each starting and ending at depot `1`.
2. Every client `2:(inst.n + 1)` is visited exactly once.
3. Every route respects the vehicle capacity `inst.Q`.
Returns `(valid = Bool, cost = Float64, loads = Vector{Int}, message = String)`.
"""
function validate_solution(
    inst::CVRPInstance,
    routes::Vector{Vector{Int}};
    atol::Float64 = 1e-5,
)
    if length(routes) != inst.K
        return (valid = false, cost = Inf, loads = Int[],
                message = "Expected $(inst.K) routes, got $(length(routes))")
    end

    visited = zeros(Int, inst.n + 1)
    loads = zeros(Int, length(routes))
    total_cost = 0.0

    for (r_idx, r) in enumerate(routes)
        if length(r) < 3 || r[1] != 1 || r[end] != 1
            return (valid = false, cost = Inf, loads = loads,
                    message = "Route $r_idx does not start/end at depot 1: $r")
        end
        for pos in 1:(length(r) - 1)
            u, v = r[pos], r[pos + 1]
            total_cost += inst.C[u, v]
            if pos >= 2
                visited[u] += 1
                loads[r_idx] += inst.q[u]
            end
        end
        if loads[r_idx] > inst.Q
            return (valid = false, cost = total_cost, loads = loads,
                    message = "Route $r_idx exceeds capacity: $(loads[r_idx]) > $(inst.Q)")
        end
    end

    for c in 2:(inst.n + 1)
        if visited[c] != 1
            return (valid = false, cost = total_cost, loads = loads,
                    message = "Client $(c - 1) (node $c) visited $(visited[c]) times")
        end
    end

    return (valid = true, cost = total_cost, loads = loads, message = "OK")
end
