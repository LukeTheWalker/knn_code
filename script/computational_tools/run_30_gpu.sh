#!/bin/bash -l

# Start doing for every cluster the baseline, if not already done
# Create the folder results and go inside
#################################################

rm -rf script/computational_tools/results_testbench
mkdir -p script/computational_tools/results_testbench
cd script/computational_tools/results_testbench

for i in {1..30}
    do
    mkdir result_$i
    cd result_$i
    sbatch ../../make_gpu.sh
    cd ..
done