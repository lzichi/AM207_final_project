#!/bin/bash
#SBATCH -J run_ace_valid
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00-00:10
#SBATCH -p kozinsky,sapphire,seas_compute
#SBATCH -e logs/%x_%j.err       
#SBATCH -o logs/%x_%j.out

source ~/.bashrc
micromamba activate dadapy_test

# Array of Z values
zs=(3.5 5.0)
for z in "${zs[@]}"; do
        sbatch <<EOT
#!/bin/bash
#SBATCH -J dpa_inter_21120_${z}
#SBATCH -N 1
#SBATCH --cpus-per-task=1 --ntasks=64
#SBATCH -t 00-72:00
#SBATCH -p sapphire,seas_compute
#SBATCH --mem-per-cpu=10GB
#SBATCH -e logs/%x_%j.err       
#SBATCH -o logs/%x_%j.out

module load intel-mkl/24.0.1-fasrc01 gcc/12.2.0-fasrc01 openmpi/4.1.4-fasrc01 cmake/3.25.2-fasrc01

source ~/.bashrc
micromamba activate dadapy_test
OMP_NUM_THREADS=64 python -u ACE_fast.py  ${z}
EOT
done