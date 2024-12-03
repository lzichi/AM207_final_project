#!/bin/bash
#SBATCH -J ace_torch
#SBATCH --cpus-per-task=2 --ntasks=32 
#SBATCH -N 1
#SBATCH -t 00-10:00
#SBATCH -p test,sapphire,seas_compute,kozinsky
#SBATCH --no-requeue
#SBATCH -e logs/%x_%j.err       
#SBATCH -o logs/%x_%j.out
#SBATCH --mem=100GB

module load intel-mkl/24.0.1-fasrc01 gcc/12.2.0-fasrc01 openmpi/4.1.4-fasrc01 cmake/3.25.2-fasrc01

source ~/.bashrc
micromamba activate flarelite
OMP_NUM_THREADS=32 python -u ACE.py 4 3 2