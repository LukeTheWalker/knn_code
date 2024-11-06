#!/bin/bash -l

# Start doing for every cluster the baseline, if not already done
# Create the folder results and go inside
#################################################

#ATTENTIOONNNNNNNNNNNNNNNNNNNNNNNNNNNNN
#FIRST DO THE FIRST, THEN THE SECOND



if [ "$1" == "y" ]; then
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


if [ "$2" == "y" ]; then
    cd results_testbench
    touch data.csv
    csv_file="data.csv"
    echo "dim,time" >> "$csv_file"

    for i in 2000 4000 8000 10000 16000 32000 64000 100000 128000
        do
        cp result_$i/baseline.txt baseline.txt
        cat "baseline.txt" >> "$csv_file"
        rm baseline.txt
    done

    mv data.csv ../../../data_iris/baseline

    cd ..

    rm -r results_testbench
fi