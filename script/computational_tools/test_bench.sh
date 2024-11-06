#!/bin/bash -l

# Start doing for every cluster the baseline, if not already done
# Create the folder results and go inside

if "$1" == "y";
then
    mkdir results_testbench
    cd results_testbench
    for i in 2000 4000 8000 10000 16000 32000 64000 100000 128000
        do
        mkdir result_$i
        cd result_$i
        sbatch ../../make_baseline.sh $i
        cd ..
    done
fi

if ["$2" == "y"];
then
    csv_file="data.csv"

    for i in 2000 4000 8000 10000 16000 32000 64000 100000 128000
        do
        echo "$i" >> "$csv_file" 
    done
fi