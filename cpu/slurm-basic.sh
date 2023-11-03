#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J FirstSlurm
#SBATCH -o FirstSlurm.out
#SBATCH -e FirstSlurm.err
#SBATCH -t 0-00:00:30
#SBATCH -p instruction
hostname