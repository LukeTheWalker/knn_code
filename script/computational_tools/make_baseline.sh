#!/bin/bash -l
#SBATCH -J knn_batch_job        # Job name
#SBATCH -N 1                    # Number of nodes
#SBATCH --ntasks=1              # Total number of MPI processes
#SBATCH --output=baseline.txt # Name of output file
#SBATCH --error=base_error.txt # Name of output file
#SBATCH --time=10:00:00         # Time to execute the script
#SBATCH -p batch                # Partition

# Load modulues
module load mpi/OpenMPI

# Load virtual environment
micromamba activate PML 

# Run your code
for i in 128000 100000 64000 32000 16000 10000 8000 4000 2000
do
    python3 ../../../../src/baseline/test_knn.py $i 500 2
done