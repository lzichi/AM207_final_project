#!/bin/bash
#SBATCH -J process_data
#SBATCH -n 64
#SBATCH -t 00-02:00
#SBATCH -p test,sapphire,seas_compute,kozinsky
#SBATCH --mem=100GB
#SBATCH -e logs/%x_%j.err       
#SBATCH -o logs/%x_%j.out

source ~/.bashrc
micromamba activate myenv
python process_data.py
