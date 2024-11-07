#!/bin/bash -l

# Start doing for every cluster the baseline, if not already done
# Create the folder results and go inside
#################################################
echo prova
cd script/computational_tools/results_testbench
echo cacca
for i in {1..30}
    do
    mkdir result_$i
    cd result_$i
    sbatch ../../make_baseline.sh
    cd ..
done