# Dynamic Scheduling & Heuristics in Semiconductor Photolithography

A simulation and optimization framework in Python addressing a complex dynamic scheduling problem inspired by a **photolithography bay in semiconductor manufacturing**.

## Problem Formulation
The model simulates an area with parallel processing units subject to shared and constrained auxiliary resources:
- **Parallel Machines ($M = 4$):** Dynamic job arrivals with non-zero release times ($r_j$).
- **Constrained Auxiliary Resources ($R = 5$):** Each job requires an exposure reticle that can only be assigned to one machine at a time.
- **Anticipatory Setups ($S = 10$):** Reticle swaps require a setup operation that can begin in anticipation as soon as both the machine and reticle become available.
- **Machine Eligibility:** Heterogeneous compatibility between wafers and lithography tracks.
- **Objective:** Minimizing total weighted tardiness ($\sum w_j T_j$), while monitoring makespan ($C_{\max}$) and tardy jobs.

## Implemented Algorithms
1. **Priority Dispatching Rules:** SPT, LPT, EDD, WSPT, ERD, MDD, WMDD.
2. **Local Search & Multistart:** Neighborhood search based on pairwise job swaps.
3. **Metaheuristics:**
   - **Simulated Annealing:** Temperature decay with exponential acceptance probability.
   - **Tabu Search:** Short-term memory tenure avoiding cyclic non-improving states.
   - **Genetic Algorithm:** Order Crossover (OX), inversion mutation, and tournament selection.

## Reporting & Output
The script automatically exports an exhaustive multi-sheet Excel dashboard (`xlsxwriter`) featuring:
- KPI summary tables comparing all algorithms.
- Metaheuristic convergence tracking plots.
- Granular job-level breakdown and Gantt timelines detailing reticle transitions and setups.

## Getting Started
```bash
git clone [https://github.com/](https://github.com/)<tuo-username>/semiconductor-photolithography-scheduling.git
cd semiconductor-photolithography-scheduling
pip install -r requirements.txt
python main.py