import numpy as np
import time
import os
import sys
import pandas as pd
from io import StringIO
import statistics

PATH_DATA_AION = os.environ["SCRATCH"] + "knn_code/data_aion/"
PATH_SCRIPT = os.environ["SCRATCH"] + "knn_code/script/"
PATH_BASE_FOLDER = os.environ["SCRATCH"] + "knn_code/"

if sys.argv[1] == "base-computation":
    print("Compute the baseline")
    os.system(PATH_SCRIPT + "computational_tools/run_30.sh")


if sys.argv[1] == "base-gen":
    print("AAA")
    file = "rows,time\n"
    for i in range(30):
        fileOpen = open(PATH_SCRIPT + "computational_tools/results_testbench/result_" + str(i+1) +"/baseline.txt", "r")
        file += fileOpen.read()

    file = StringIO(file)
    df = pd.read_csv(file, sep=",")
    df = df.sort_values("rows")

    _mean = []
    _std = []
    rows = [2000,4000,8000,10000,16000,32000,64000,100000,128000]
    
    for i in range(9):
        _mean.append(statistics.mean(df['time'][i*30:(i+1)*30]))
        _std.append(statistics.stdev(df['time'][i*30:(i+1)*30]))

    dict = {'rows': rows, 'mean': _mean, 'std': _std} 
    df2 = pd.DataFrame(dict)
    print(df2)
    df2.to_csv(PATH_DATA_AION + '/baseline/data.csv', sep = ',', index=False)
    os.system("rm -r " + PATH_SCRIPT + "computational_tools/results_testbench")
    os.system("mkdir " + PATH_SCRIPT + "computational_tools/results_testbench")
