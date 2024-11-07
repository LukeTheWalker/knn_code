#!/bin/bash -l
#SBATCH -J knn_batch_job        # Job name
#SBATCH -N 1                    # Number of nodes
#SBATCH --ntasks=14             # Total number of MPI processes
#SBATCH --cpus-per-task=2
#SBATCH --output=mpi.txt        # Name of output file
#SBATCH --error=mpi_error.txt  # Name of output file
#SBATCH --time=10:00:00         # Time to execute the script
#SBATCH -p batch                # Partition

# Load modulues
module load mpi/OpenMPI

# Load virtual environment
micromamba activate PML 

# Run your code
mpiexec -n $1 test_knn.py $2 500 1000