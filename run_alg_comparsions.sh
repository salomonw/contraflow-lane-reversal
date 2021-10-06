#!/bin/bash -l

#$ -P noc-lab
#$ -N Contraflow
#$ -j y
#$ -m bae
#$ -pe omp 8
#$ -l h_rt=17:59:00

# Load Modules
module load gurobi

# Define Network
nets="test_9 EMA EMA_mid Anaheim NYC"
#nets="NYC"
# Define Demand multiplier
g_mult="3"

for net in $nets; do
    # Run programs
    python3 -m experiments.run_algorithm_comparison $net $g_mult
done

