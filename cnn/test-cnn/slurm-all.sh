#!/usr/bin/env zsh

#SBATCH --job-name=Task1
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:10:00
#SBATCH --error=error-%j.txt
#SBATCH --output=output-%j.txt

module load nvidia/cuda/11.8.0
module load gcc/.11.3.0_cuda

cd $SLURM_SUBMIT_DIR

make clean
make 

if [ -f "test-cnn" ]; then
	echo "test-cnn created"
fi

echo "Running test-cnn"

# echo "CPU mnist-small n_images 400"
# ./test-cnn cpu 13 0.1 100 1 400

# echo "CPU mnist-small n_images 2000"
# ./test-cnn cpu 13 0.1 100 1 2000

# echo "CPU mnist-small n_images 10000"
# ./test-cnn cpu 13 0.1 100 1 10000

echo "GPU naive mnist-small n_images 400"
./test-cnn gpu 13 0.1 10 1 400

echo "GPU naive mnist-small n_images 2000"
./test-cnn gpu 13 0.1 10 1 2000

echo "GPU naive mnist-small n_images 10000"
./test-cnn gpu 13 0.1 10 1 10000

