import matplotlib.pyplot as plt
import re

# Define a function to parse the log data
def parse_log_data(file_path):
    # Regular expressions to extract the number of MPI processes and time
    patternTime = r"time (\d+\.\d+)"
    patternMPI = r"MPI processes: (\d+)"

    mpi_processes = []
    times = []
    
    # Open the file and read the data
    with open(file_path, 'r') as file:
        log_data = file.read()
    
    # Find all matches in the log data
    for match in re.finditer(patternMPI, log_data):
        mpi_processes.append(int(match.group(1)))  # MPI processes  # Time

    for match in re.finditer(patternTime, log_data):
        times.append(float(match.group(1)))       # Time
    
    return mpi_processes, times

# Set the path to your text file (replace with your actual file path)
file_path = "test_knn.txt"  # Change this to the path of your log file

# Parse the data from the file
mpi_processes, times = parse_log_data(file_path)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(mpi_processes, times, marker='o', linestyle='-', color='b', label='Execution Time')

# Add labels and title
plt.xlabel('Number of MPI Processes')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Number of MPI Processes')
plt.grid(True)
plt.legend()

# Show the plot
plt.savefig("prova.png")
