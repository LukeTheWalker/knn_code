import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os

graph_folder = 'graphs'

if not os.path.exists(graph_folder):
    os.makedirs(graph_folder)

number_of_run = 5

tests = {
    'baseline': {
        'active': True,
        'processors': [1],
        'command': 'python baseline/test_knn.py {rows} {cols} {k}'
    },
    'gpu': {
        'active': False,
        'processors': [i for i in range(1,5)],
        'command': 'python test_knn.py {rows} {cols} {k}'
    },
    'mpi_core': {
        'active': True,
        'processors': [2, 4, 8, 16, 32, 64, 128],
        'command': 'mpirun -np {processors} python mpi/test_knn.py {rows} {cols} {k}'
        # 'command': 'srun -n {processors} python mpi/test_knn.py {rows} {cols} {k}'
    },
    'mpi_node': {
        'active': False,
        'processors': [1,2,3,4],
        'command': 'mpirun -np {processors} -H localhost python test_knn.py {rows} {cols} {k}'
    }
}

class Timing:
    def __init__(self, times):
        self.times = times
        self.mean = np.mean(times)
        self.std = np.std(times)


class TestSuite:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cols = 500
        self.k = 5
        self.baseline_times_df = None
        self.baseline_load()

    def baseline_load(self):
        # if the cache file is older than the last modified file in baseline, then run the baseline test
        if self.baseline_cache_out_of_date():
            self.baseline_times_df = pd.DataFrame(columns=['size', 'mean', 'std'])
        else:
            self.baseline_times_df = pd.read_csv(self.cache_file, header=0)

    def baseline_save(self):
        self.baseline_times_df.to_csv(self.cache_file, index=False)

    def baseline_cache_out_of_date(self):
        # if the cache file is older than the last modified file in baseline, then run the baseline test
        if not os.path.exists(self.cache_file):
            return True
        # check also if the cache file is older than this script
        # if os.path.getmtime(self.cache_file) < os.path.getmtime(__file__):
        #     return True
        last_modified = os.path.getmtime(self.cache_file)
        for root, _, files in os.walk('baseline'):
            for file in files:
                if file.endswith('.py'):
                    if os.path.getmtime(os.path.join(root, file)) > last_modified:
                        return True
        return False


    def baseline_test(self, rows, cols, k):
        print(f'Running baseline test for size {rows}')
        # if the cache dataframe has the processors, return the mean and std
        print(list(self.baseline_times_df['size']))
        if rows in list(self.baseline_times_df['size']):
            t = Timing([0,0]);
            t.mean = self.baseline_times_df.loc[self.baseline_times_df['size'] == rows, 'mean'].values[0]
            t.std  = self.baseline_times_df.loc[self.baseline_times_df['size'] == rows,  'std'].values[0]
            print(f'Loaded from cache: {t.mean} +- {t.std}')
            return t
        else:
            time_run = []
            for _ in range(number_of_run):
                start_time = time.time()
                command = tests['baseline']['command'].format(rows=rows, cols=cols, k=k)
                os.system(command)
                time_run.append(time.time() - start_time)
            timing = Timing(time_run)
            new_row = pd.DataFrame({'size': [rows], 'mean': [timing.mean], 'std': [timing.std]})
            self.baseline_times_df = new_row if self.baseline_times_df.empty else pd.concat([self.baseline_times_df, new_row], ignore_index=True)
            self.baseline_save()
            print(f'Baseline added: {timing.mean} +- {timing.std}')
            return timing

    def standard_analysis(self):
        rows = 10000
        times = {}
        print("------------------------------------------------")
        print('Starting standard Analysis')
        times['baseline'] = self.baseline_test(rows, self.cols, self.k)
        print(f'Baseline: {times["baseline"].mean} +- {times["baseline"].std}')

        for test_name, test in tests.items():
            if test_name == 'baseline' or not test['active']:
                continue
            p = max(test['processors'])
            time_run = []
            for _ in range(number_of_run):
                start_time = time.time()
                command = test['command'].format(rows=rows, cols=self.cols, k=self.k, processors=p)
                os.system(command)
                time_run.append(time.time() - start_time)
            times[test_name] = Timing(time_run)
            print(f'{test_name} {p} processors: {times[test_name].mean} +- {times[test_name].std}, speedup: {times["baseline"].mean/times[test_name].mean}')

        print('Finished standard Analysis')

    def strong_scaling(self):
        rows = 10000

        print("------------------------------------------------")
        print('Starting strong scaling')

        times = {}
        times['baseline'] = self.baseline_test(rows, self.cols, self.k)

        results = []

        for test_name, test in tests.items():
            if test_name == 'baseline' or not test['active']:
                continue
            for n in test['processors']:
                time_run = []
                for _ in range(number_of_run):
                    start_time = time.time()
                    command = test['command'].format(rows=rows, cols=self.cols, k=self.k, processors=n)
                    os.system(command)
                    time_run.append(time.time() - start_time)
                times[f'{test_name}_{n}'] = Timing(time_run)
                results.append({
                    'test_name': test_name,
                    'processors': n,
                    'mean_time': times[f'{test_name}_{n}'].mean,
                    'std_time': times[f'{test_name}_{n}'].std,
                    'speedup': times['baseline'].mean / times[f'{test_name}_{n}'].mean
                })

        df = pd.DataFrame(results)
        df.to_csv('strong_scaling_results.csv', index=False)

        # Create graphs
        fig, ax = plt.subplots()
        processors = []
        speedups = []
        for test_name, test in tests.items():
            if test_name == 'baseline' or not test['active']:
                continue
            for n in test['processors']:
                processors.append(n)
                speedups.append(times['baseline'].mean / times[f'{test_name}_{n}'].mean)
                ax.errorbar(n, times[f'{test_name}_{n}'].mean, yerr=times[f'{test_name}_{n}'].std, fmt='o', label=f'{test_name} {n} processors')

        ax.set_xlabel('Number of Processors')
        ax.set_ylabel('Execution Time (s)')
        ax.set_title('Strong Scaling Performance')
        ax.legend()
        plt.savefig(f'{graph_folder}/strong_scaling_performance.png')

        # Plot speedup
        fig, ax = plt.subplots()
        ax.plot(processors, speedups, 'o-')
        ax.set_xlabel('Number of Processors')
        ax.set_ylabel('Speedup')
        ax.set_title('Speedup vs Number of Processors')
        plt.savefig(f'{graph_folder}/speedup_vs_processors_strong_scaling.png')

        print('Finished strong scaling')

    def weak_scaling(self):
        times = {}
        rows = 1000

        print("------------------------------------------------")
        print('Starting weak scaling')

        results = []

        for test_name, test in tests.items():
            if test_name == 'baseline' or not test['active']:
                continue
            for n in test['processors']:
                time_run = []
                times[f'baseline_{n}'] = self.baseline_test(rows*n, self.cols, self.k)
                for _ in range(number_of_run):
                    start_time = time.time()
                    command = test['command'].format(rows=rows*n, cols=self.cols, k=self.k, processors=n)
                    os.system(command)
                    time_run.append(time.time() - start_time)
                times[f'{test_name}_{n}'] = Timing(time_run)
                results.append({
                    'test_name': test_name,
                    'processors': n,
                    'mean_time': times[f'{test_name}_{n}'].mean,
                    'std_time': times[f'{test_name}_{n}'].std,
                    'speedup': times[f'baseline_{n}'].mean / times[f'{test_name}_{n}'].mean
                })

        df = pd.DataFrame(results)
        df.to_csv('weak_scaling_results.csv', index=False)

        # Create graphs
        fig, ax = plt.subplots()
        processors = []
        speedups = []
        for test_name, test in tests.items():
            if test_name == 'baseline' or not test['active']:
                continue
            for n in test['processors']:
                processors.append(n)
                speedups.append(times[f'baseline_{n}'].mean / times[f'{test_name}_{n}'].mean)
                ax.errorbar(n, times[f'{test_name}_{n}'].mean, yerr=times[f'{test_name}_{n}'].std, fmt='o', label=f'{test_name} {n} processors')

        ax.set_xlabel('Number of Processors')
        ax.set_ylabel('Execution Time (s)')
        ax.set_title('Weak Scaling Performance')
        ax.legend()
        plt.savefig(f'{graph_folder}/weak_scaling_performance.png')

        # Plot speedup
        fig, ax = plt.subplots()
        ax.plot(processors, speedups, 'o-')
        ax.set_xlabel('Number of Processors')
        ax.set_ylabel('Speedup')
        ax.set_title('Speedup vs Number of Processors')
        plt.savefig(f'{graph_folder}/speedup_vs_processors_weak_scaling.png')

        print('Finished weak scaling')

if __name__ == '__main__':
    test_suite = TestSuite('baseline_times.csv')
    test_suite.standard_analysis()
    test_suite.strong_scaling()
    test_suite.weak_scaling()
    test_suite.baseline_save()