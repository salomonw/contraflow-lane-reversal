#!/bin/bash -l

#$ -P noc-lab
#$ -N Contraflow
#$ -j y
#$ -m bae
#$ -pe omp 8
#$ -l h_rt=17:59:00

# Load modules
module load gurobi

# Run programs
python3 -m experiments.test_max_reversals test_9 1
python3 -m experiments.test_max_reversals EMA_mid 2
python3 -m experiments.test_solve_net_sizes



