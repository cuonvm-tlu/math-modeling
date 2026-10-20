# Capacitated Vehicle Routing Problem (CVRP): Compact MILP Formulations & Branch-and-Bound Study

**Master of Data Science — Mathematical Modelling Capstone Project**

This repository contains the complete Julia implementation, instance generation suite, numerical benchmark pipeline, interactive `Pluto.jl` notebook, and 10-page LaTeX technical report for the **Capacitated Vehicle Routing Problem (CVRP)**, developed for the *Practical Course on Mathematical Modelling* (Prof. Viet Hung Nguyen).

---

## 1. Repository Structure

```text
capstone_project/
├── Project.toml                 # Julia package environment (JuMP, HiGHS, Bonobo, DataFrames, PlutoUI)
├── README.md                    # Project documentation and reproduction guide
├── src/
│   ├── instances.jl             # Synthetic CVRP generator (U-Center & C-Corner, 10 seeds/size) + Wolsey instance
│   ├── models.jl                # 5 compact MILP formulations in JuMP.jl (:MTZ, :LiftedMTZ, :SCF, :ThreeIndex, :MCF)
│   ├── bonobo_bnb.jl            # Custom LP-based Branch-and-Bound using Bonobo.jl
│   └── metrics.jl               # Process CPU timing (ccall(:clock)), root LP evaluator, HiGHS MIP wrapper
├── scripts/
│   ├── verify_models.jl         # Sanity check: verifies all 5 models match on Wolsey Slide 19 & synthetic instances
│   └── run_experiments.jl       # Runs 10-seed benchmark suite & exports CSV + LaTeX tables
├── notebooks/
│   └── cvrp_capstone_pluto.jl   # Interactive Pluto.jl notebook with SVG route visualization & live comparisons
├── results/                     # Generated CSV datasets and auto-exported LaTeX booktabs tables
└── report/
    └── main.tex                 # Complete 10-page LaTeX technical report following course rubric
```

---

## 2. Mathematical Formulations Implemented

Let $G = (V_0, A)$ be a complete directed graph with depot $0$ (`node 1` in 1-based Julia indexing) and client set $V = \{1, \dots, n\}$ (`nodes 2:(n+1)`), client demands $d_i > 0$, homogeneous fleet size $K$, and vehicle capacity $Q$:

| Symbol | Formulation Name | Variables | Constraints | Key Polyhedral Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| `:MTZ` | **2-Index Basic MTZ** | $O(n^2)$ bin, $O(n)$ cont | $O(n^2)$ | Cumulative load $u_i - u_j + Q x_{ij} \le Q - d_j$ |
| `:LiftedMTZ` | **2-Index Lifted MTZ** *(Desrochers & Laporte, 1991)* | $O(n^2)$ bin, $O(n)$ cont | $O(n^2)$ | Lifts reverse arc $(Q - d_i - d_j)x_{ji}$ + tightened $u_i$ bounds |
| `:SCF` | **2-Index Single-Commodity Flow** *(Gavish & Graves, 1978)* | $O(n^2)$ bin, $O(n^2)$ cont | $O(n^2)$ | Arc load $d_j x_{ij} \le f_{ij} \le (Q - d_i) x_{ij}$ (implies fractional capacity cuts) |
| `:ThreeIndex` | **3-Index Vehicle Flow** *(Wolsey, 2014, Slide 10)* | $O(K n^2)$ bin, $O(n)$ cont | $O(K n + n^2)$ | Explicit vehicle assignment $z_i^k$ + load-ordering symmetry breaking |
| `:MCF` | **2-Index Multi-Commodity Flow** *(Wolsey, 2014, Slides 7 & 26)* | $O(n^2)$ bin, $O(n^3)$ cont | $O(n^3)$ | Disaggregated client flows $f_{ij}^w \le d_w x_{ij}$ (implies both SECs & fractional capacity cuts) |

---

## 3. Reproducing the Results

### Step 1: Install Packages
From the `capstone_project` directory, instantiate the Julia environment:
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Step 2: Run Model Verification
Verify that all 5 formulations produce identical optimal integer objective values and valid routes on Laurence A. Wolsey's Lecture 6 benchmark instance ($n=3, K=2, Q=10, z^*=49$) and synthetic test instances:
```bash
julia --project=. scripts/verify_models.jl
```

### Step 3: Run the 10-Seed Numerical Experiments
Execute the full benchmark suite (Root LP Bounds, `Bonobo.jl` Branch-and-Bound tree comparison, `HiGHS` Full MIP scaling across $n \in \{10, 15, 20, 25, 30, 40\}$, and Sensitivity Analysis):
```bash
julia --project=. scripts/run_experiments.jl
```
All raw and 10-run aggregated CSV files are written to `results/`, and LaTeX tables are exported to `results/summary_tables.tex`.

### Step 4: Launch the Interactive Pluto Notebook
```bash
julia -e 'using Pkg; Pkg.add("Pluto"); import Pluto; Pluto.run(notebook="notebooks/cvrp_capstone_pluto.jl")'
```
