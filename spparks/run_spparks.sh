#!/bin/bash
#SBATCH -J spparks
#SBATCH -N 1
#SBATCH -n 8
##SBATCH --cpus-per-task=1 --ntasks=32
#SBATCH -t 00-10:00
#SBATCH -p sapphire,seas_compute,kozinsky
#SBATCH -e logs/%x_%j.err       
#SBATCH -o logs/%x_%j.out

# example of spparks input file, only change is path to .in file 
# and resources requested for each run 

source ~/.bashrc

micromamba activate /n/holystore01/LABS/kozinsky_lab/Lab/Software/SPPARKS_11_24/spparks_env
OMP_NUM_THREADS=8
module load intel-mkl/24.0.1-fasrc01 gcc/12.2.0-fasrc01 openmpi/4.1.4-fasrc01 cmake/3.25.2-fasrc01

mpirun -n 8 /n/holystore01/LABS/kozinsky_lab/Lab/Software/SPPARKS_11_24/spparks/src/spk_mpi < ../../spparks_gill_in/20ps_1ps_3.5_spparks_gill.in -log log_new.spparks