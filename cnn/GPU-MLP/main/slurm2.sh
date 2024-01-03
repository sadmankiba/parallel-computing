#!/usr/bin/env zsh
#SBATCH -c 1
#SBATCH -J nn
#SBATCH -o nn.out
#SBATCH -e nn.err
#SBATCH -t 0-00:20:00
#SBATCH -p instruction
#SBATCH --gres=gpu:1
#SBATCH --mem=40G

module load nvidia/cuda/11.8.0 gcc/.11.3.0_cuda
make clean
make
./nn

