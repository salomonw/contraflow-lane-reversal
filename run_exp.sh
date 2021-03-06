#!/bin/bash -l

#$ -P noc-lab
#$ -N Contraflow
#$ -j y
#$ -m bae
#$ -pe omp 32
#$ -l h_rt=17:59:00

# Load Modules
module load gurobi

# Define Network
#nets="test_9 EMA EMA_mid Anaheim NYC"
#nets="test_9 EMA_mid"
nets="test_9"
# Define Demand multiplier
g_mult="1"

for net in $nets; do
    # Run programs
    python3 -m experiments.test_max_reversals $net $g_mult
    #python3 -m experiments.test_lambda_sensitivity $net $g_mult
    python3 -m experiments.run_od_benefit $net $g_mult
    #python3 -m experiments.run_algorithm_comparison $net $g_mult
    #python3 -m experiments.run_restrict_num_lanes $net $g_mult
    python3 -m experiments.run_demand_levels $net $g_mult
    # Plot results
    python3 -m experiments.run_format_results_journal $net $g_mult
done



