#!/bin/bash -l

#$ -P noc-lab
#$ -N Contraflow
#$ -j y
#$ -m bae
#$ -pe omp 8
#$ -l h_rt=11:59:00

# Load Modules
module load python3
module load gurobi

# Run program
python3 -m experiments.test_contraflow