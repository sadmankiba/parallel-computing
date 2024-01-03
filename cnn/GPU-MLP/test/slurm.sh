#!/usr/bin/env zsh
#SBATCH -c 1
#SBATCH -J test
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH -t 0-00:15:00
#SBATCH -p instruction
#SBATCH --gres=gpu:1
#SBATCH --mem=20G

module load nvidia/cuda/11.8.0 gcc/.11.3.0_cuda
make clean
make
./unit
./model

