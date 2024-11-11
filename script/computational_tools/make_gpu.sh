#!/bin/bash -l
#SBATCH -J knn_batch_job        # Job name
#SBATCH -N 1                    # Number of nodes
#SBATCH --ntasks=4              # Total number of MPI processes
#SBATCH --output=gpu_out.txt    # Name of output file
#SBATCH --error=gpu_err.txt     # Name of output file
#SBATCH --time=10:00:00         # Time to execute the script
#SBATCH --partition=gpu         # Partition
#SBATCH --gres=gpu:4            # Number of GPUs

# Load virtual environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate cuda_env 

export PATH="/home/users/lgreco/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/home/users/lgreco/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME=/home/users/lgreco/cuda

# Run your code
for i in 128000 100000 64000 32000 16000 10000 8000 4000 2000
do
    python3 ../../../../src/gpu/test_knn.py $i 500 2
done