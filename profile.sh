#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Error: You must provide exactly two arguments."
    echo "Usage: ./profile.sh <num_nodes> <num_edges>"
    exit 1
fi

make clean
make
echo "Generating input"
python3 src/generate_input.py $1 $2 input.txt
echo "Running pagerank-cpp"
time ./bin/pagerank-cpp input.txt outputs/output_cpp.txt
echo "Running pagerank-optimised"
time ./bin/pagerank-opt input.txt outputs/output_optimised.txt
echo "Running pagerank-cuda"
time ./bin/pagerank-cuda input.txt outputs/output_cuda.txt