#!/usr/bin/env bash

# Usage: ./profile.sh
#
# This script runs PageRank computations for various graph sizes and densities,
# comparing CPU-naive, CPU-optimized, and CUDA implementations.
#
# It now includes three sets of (N, E) pairs:
# - Standard Set: moderate densities
# - Dense Set: higher densities
# - Highly Sparse Set: extremely low density
#
# Pre-conditions:
# - src/generate_input.py is available and can generate graph files.
# - ./bin/pagerank-cpp, ./bin/pagerank-opt, ./bin/pagerank-cuda are compiled and ready.
# - "make clean && make" rebuilds these binaries.

if [ ! -f "src/generate_input.py" ]; then
    echo "Error: generate_input.py not found in src directory."
    exit 1
fi

if [ ! -d "outputs" ]; then
    mkdir outputs
fi

if [ ! -d "bin" ]; then
    mkdir bin
fi

# Define parameter sets
# Highly sparse set: extremely low edge counts relative to nodes, e.g., ~N edges or less
# Example: For N=100,000 and E=50,000 edges, sparsity = 5e4 / 1e10 = 5e-6, very low
HIGHLY_SPARSE_NODES_ARR=(100000 500000 1000000)
HIGHLY_SPARSE_EDGES_ARR=(500000 2000000 5000000)

make clean
make

RESULTS_FILE="results_bigger.txt"

echo "Starting Profile Runs" > $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "-------------------------------------" >> $RESULTS_FILE

run_tests() {
    local NODE_ARR=("$1")
    local EDGE_ARR=("$2")
    local LABEL="$3"

    # Because we pass arrays as strings, we need to re-parse them
    # We'll rely on the caller providing the arrays inline.
    # Example call: run_tests "${STANDARD_NODES_ARR[@]}" "${STANDARD_EDGES_ARR[@]}" "STANDARD SET"
    # $1 would be first node, $2 first edge, etc. We'll handle them differently:
    NODE_ARR=("$@") # re-parse all arguments
    # We know last argument is the label
    LABEL=${NODE_ARR[-1]}
    # Extract label and arrays
    # The label might have spaces, so handle carefully
    # We'll find the label index first:
    local label_idx=${#NODE_ARR[@]}
    ((label_idx--))

    LABEL="${NODE_ARR[$label_idx]}"
    unset 'NODE_ARR[$label_idx]'

    # Half of the remaining are NODES_ARR and half are EDGES_ARR
    local half=$(( ${#NODE_ARR[@]} / 2 ))
    local NODES=("${NODE_ARR[@]:0:$half}")
    local EDGES=("${NODE_ARR[@]:$half:$half}")

    echo "" >> $RESULTS_FILE
    echo "===== BEGIN $LABEL =====" | tee -a $RESULTS_FILE
    for N in "${NODES[@]}"; do
        for E in "${EDGES[@]}"; do
            echo "" >> $RESULTS_FILE
            echo "Running tests for N=${N}, E=${E} [${LABEL}]" | tee -a $RESULTS_FILE

            INPUT_FILE="input_${N}_${E}.txt"

            echo "Generating input with N=$N, E=$E"
            python3 src/generate_input.py $N $E $INPUT_FILE

            # Compute sparsity = E / (N^2)
            SPARSITY=$(awk "BEGIN {printf \"%.8e\", $E/($N*$N)}")
            echo "Parameters: N=$N, E=$E, Sparsity=$SPARSITY" | tee -a $RESULTS_FILE

            # Run Optimized CPU
            echo "Running pagerank-opt (Optimized CPU)..." | tee -a $RESULTS_FILE
            START_TIME=$(date +%s%3N)
            ./bin/pagerank-opt $INPUT_FILE "outputs/output_optimised_${N}_${E}.txt" >> $RESULTS_FILE 2>&1
            END_TIME=$(date +%s%3N)
            ELAPSED=$((END_TIME-START_TIME))

            # Run CUDA
            echo "Running pagerank-cuda (CUDA)..." | tee -a $RESULTS_FILE
            START_TIME=$(date +%s%3N)
            ./bin/pagerank-cuda $INPUT_FILE "outputs/output_cuda_${N}_${E}.txt" >> $RESULTS_FILE 2>&1
            END_TIME=$(date +%s%3N)
            ELAPSED=$((END_TIME-START_TIME))

            echo "------------------------------------------------" >> $RESULTS_FILE
        done
    done
    echo "===== END $LABEL =====" | tee -a $RESULTS_FILE
}


# Run tests for the three sets
run_tests "${HIGHLY_SPARSE_NODES_ARR[@]}" "${HIGHLY_SPARSE_EDGES_ARR[@]}" "HIGHLY SPARSE SET"

echo "All runs completed. Results stored in $RESULTS_FILE."
