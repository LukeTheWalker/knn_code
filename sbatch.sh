#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --ntasks-per-socket=16
#SBATCH -c 1
#SBATCH --job-name=knn_computation
#SBATCH --output=knn_computation.out
#SBATCH --error=knn_computation.err
#SBATCH --time=00:10:00

#4090990

# Load necessary modules
cd /home/users/lgreco/Development/ML4HPC/knn_code
module load mpi/OpenMPI/4.0.5-GCC-10.2.0 lang/Python/3.8.6-GCCcore-10.2.0
source venv/bin/activate
module load mpi/OpenMPI/4.0.5-GCC-10.2.0

# Run your computation
mpirun -np 128 python -m cProfile -s tottime mpi/test_knn.py 100000 500 5