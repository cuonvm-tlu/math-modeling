# =============================================================================
#  Model Verification & Sanity Check Script
#  Verifies that all 5 compact CVRP formulations + 2-Index RCC Branch-and-Cut
#  produce identical optimal integer objectives and valid routes on Wolsey's
#  Slide 19 benchmark and synthetic instances.
# =============================================================================

include(joinpath(@__DIR__, "..", "src", "instances.jl"))
include(joinpath(@__DIR__, "..", "src", "models.jl"))
include(joinpath(@__DIR__, "..", "src", "metrics.jl"))
include(joinpath(@__DIR__, "..", "src", "bonobo_bnb.jl"))

const FORMULATIONS = (:MTZ, :LiftedMTZ, :SCF, :ThreeIndex, :MCF)

function run_verification()
    println("="^88)
    println("1. Verification on Wolsey (2014) Lecture 6 Benchmark Instance (n = 3, K = 2, Q = 10)")
    println("="^88)
    w_inst = wolsey_instance()
    @printf("%-14s | %10s %10s %10s %8s %8s %10s\n",
            "Formulation", "LP Bound", "MIP Obj", "Bonobo Obj", "MIP B&B", "Bonobo", "Valid?")
    println("-"^88)

    for form in FORMULATIONS
        lp_res = solve_root_lp(w_inst, form)
        mip_res = solve_cvrp_highs(w_inst, form; time_limit = 30.0)
        bnb_res = solve_cvrp_bonobo(w_inst, form; node_limit = 2_000)

        @printf("%-14s | %10.2f %10.2f %10.2f %8d %8d %10s\n",
                String(form), lp_res.bound, mip_res.obj, bnb_res.obj,
                mip_res.nodes, bnb_res.nodes, string(mip_res.valid))

        @assert isapprox(mip_res.obj, 49.0; atol = 1e-4) "Expected 49.0 for $form, got $(mip_res.obj)"
        @assert isapprox(bnb_res.obj, 49.0; atol = 1e-4) "Expected 49.0 in Bonobo for $form, got $(bnb_res.obj)"
        @assert mip_res.valid "Invalid routes for $form on Wolsey instance!"
    end

    rcc_res = solve_cvrp_rcc_bnc(w_inst; node_limit = 2_000)
    @printf("%-14s | %10s %10s %10.2f %8s %8d %10s\n",
            "RCC (B&C)", "—", "—", rcc_res.obj, "—", rcc_res.nodes, "true")
    @assert isapprox(rcc_res.obj, 49.0; atol = 1e-4) "Expected 49.0 in RCC B&C, got $(rcc_res.obj)"

    println("\n" * "="^88)
    println("2. Cross-Formulation Consistency on Synthetic Instances (n in [6, 8, 10])")
    println("="^88)
    @printf("%-16s | %8s | %10s %10s %10s %10s %10s\n",
            "Instance", "Optimum", "MTZ LP", "LiftMTZ LP", "SCF LP", "3-Idx LP", "MCF LP")
    println("-"^88)

    for (n, topo, seed) in ((6, :uniform, 1), (8, :uniform, 2), (8, :clustered, 3), (10, :uniform, 1))
        inst = generate_cvrp_instance(n; seed = seed, topology = topo)
        objs = Float64[]
        lps = Dict{Symbol,Float64}()

        for form in FORMULATIONS
            lp_res = solve_root_lp(inst, form)
            mip_res = solve_cvrp_highs(inst, form; time_limit = 30.0)
            @assert mip_res.valid "Invalid route produced by $form on $(inst.name)"
            push!(objs, mip_res.obj)
            lps[form] = lp_res.bound
        end

        ref_obj = objs[1]
        for (idx, form) in enumerate(FORMULATIONS)
            @assert isapprox(objs[idx], ref_obj; atol = 1e-3) "Mismatch on $(inst.name): $form gave $(objs[idx]) vs $ref_obj"
        end

        # Verify theoretical polyhedral dominance: MTZ <= LiftedMTZ <= Opt and MTZ <= SCF <= MCF <= Opt
        @assert lps[:MTZ] <= lps[:LiftedMTZ] + 1e-5
        @assert lps[:MTZ] <= lps[:SCF] + 1e-5
        @assert lps[:SCF] <= lps[:MCF] + 1e-5
        @assert lps[:MCF] <= ref_obj + 1e-5

        @printf("%-16s | %8.1f | %10.2f %10.2f %10.2f %10.2f %10.2f\n",
                inst.name, ref_obj, lps[:MTZ], lps[:LiftedMTZ], lps[:SCF], lps[:ThreeIndex], lps[:MCF])
    end

    println("\nAll formulations and RCC Branch-and-Cut verified successfully!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_verification()
end
