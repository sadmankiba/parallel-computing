#!/usr/bin/env zsh
#SBATCH -c 1
#SBATCH -J nn
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH -t 0-00:20:00
#SBATCH -p instruction
#SBATCH --mem=30G

module load nvidia/cuda/11.8.0 gcc/.11.3.0_cuda

make clean
make
./test.out