import pandas as pd
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import ScalarFormatter

def strong_scaling(data_baseline, data_test, graph_folder):
    rows = 10000

    print("------------------------------------------------")
    print('Starting strong scaling')

    # Load data, columns are rows,mean,std
    baseline = pd.read_csv(data_baseline, header=0)
    test = pd.read_csv(data_test, header=0)

    # remove data rows == 10000
    baseline = baseline[baseline['rows'] != 10000]
    test = test[test['rows'] != 10000]

    # Compute speedup
    speedup = baseline['mean'] / test['mean']

    # Plot speedup in log-log scale
    plt.figure(figsize=(15, 8))
    plt.plot(test['rows'], speedup, marker='o', label='Speedup')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of rows (log scale)')
    plt.ylabel('Speedup (log scale)')
    plt.title('Strong scaling (log-log scale)')
    plt.grid()
    plt.legend()

    # Set custom ticks
    plt.xticks(test['rows'])
    plt.yticks(speedup)

    # Avoid scientific notation for xticks
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)

    # Avoid scientific notation for yticks
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)

    plt.savefig(graph_folder + 'speedup_loglog.png')
    plt.close()

    print('Strong scaling completed')

    return speedup

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python speedup.py <data_baseline> <data_test> <graph_folder>')
        sys.exit(1)
    strong_scaling(sys.argv[1], sys.argv[2], sys.argv[3])
