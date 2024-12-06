#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr double DAMPING_FACTOR = 0.85;

__global__ void pagerank_kernel(int num_nodes, int *row_offsets,
                                int *col_indices, int *out_degrees,
                                double *rank_old, double *rank_new,
                                double dangling_sum) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < num_nodes) {
    double sum = dangling_sum / num_nodes;
    int row_start = row_offsets[i];
    int row_end = row_offsets[i + 1];
    for (int idx = row_start; idx < row_end; ++idx) {
      int j = col_indices[idx]; // Node j links to node i
      int out_degree = out_degrees[j];
      if (out_degree > 0) {
        sum += rank_old[j] / out_degree;
      }
    }
    rank_new[i] = (1.0 - DAMPING_FACTOR) / num_nodes + DAMPING_FACTOR * sum;
  }
}

__global__ void compute_dangling_sum(int num_nodes, int *out_degrees,
                                     double *rank_old,
                                     double *dangling_contrib) {
  __shared__ double sdata[256];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  double my_sum = 0.0;
  if (i < num_nodes && out_degrees[i] == 0) {
    my_sum = rank_old[i];
  }
  sdata[tid] = my_sum;
  __syncthreads();

  // Reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Write result from each block to global memory
  if (tid == 0) {
    atomicAdd(dangling_contrib, sdata[0]);
  }
}

void build_graph(const std::string &filepath, int &num_nodes, int &num_edges,
                 std::vector<int> &row_offsets, std::vector<int> &col_indices,
                 std::vector<int> &out_degrees, std::vector<int> &node_ids) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file");
  }
  std::string line;
  std::unordered_map<int, int> node_id_to_index;
  num_nodes = 0;
  num_edges = 0;

  std::vector<std::vector<int>> adj_incoming;
  std::vector<int> out_degrees_temp;

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string token;
    std::getline(ss, token, ',');
    int from = std::stoi(token);
    std::getline(ss, token, ',');
    int to = std::stoi(token);

    // Map node IDs to indices
    if (node_id_to_index.find(from) == node_id_to_index.end()) {
      node_id_to_index[from] = num_nodes++;
      node_ids.push_back(from);
      adj_incoming.push_back(std::vector<int>());
      out_degrees_temp.push_back(0);
    }
    if (node_id_to_index.find(to) == node_id_to_index.end()) {
      node_id_to_index[to] = num_nodes++;
      node_ids.push_back(to);
      adj_incoming.push_back(std::vector<int>());
      out_degrees_temp.push_back(0);
    }

    int from_idx = node_id_to_index[from];
    int to_idx = node_id_to_index[to];

    // Build adjacency list of incoming edges
    adj_incoming[to_idx].push_back(from_idx);
    num_edges++;

    // Increment out-degree of 'from' node
    out_degrees_temp[from_idx]++;
  }

  // Build CSR arrays
  row_offsets.resize(num_nodes + 1);
  row_offsets[0] = 0;
  for (int i = 0; i < num_nodes; ++i) {
    row_offsets[i + 1] = row_offsets[i] + adj_incoming[i].size();
  }

  col_indices.resize(num_edges);
  int idx = 0;
  for (int i = 0; i < num_nodes; ++i) {
    for (int j = 0; j < adj_incoming[i].size(); ++j) {
      col_indices[idx++] = adj_incoming[i][j];
    }
  }

  out_degrees = out_degrees_temp;
}

void write_rank(const std::vector<int> &node_ids,
                const std::vector<double> &rank, const std::string &filepath) {
  std::ofstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file");
  }
  // Create a vector of indices
  std::vector<int> indices(rank.size());
  for (int i = 0; i < rank.size(); ++i) {
    indices[i] = i;
  }
  // Sort indices based on rank
  std::sort(indices.begin(), indices.end(),
            [&rank](int a, int b) { return rank[a] > rank[b]; });
  for (const auto &idx : indices) {
    file << node_ids[idx] << "," << rank[idx] << std::endl;
  }
}

int main(int argc, char *argv[]) {
  std::string input_file = std::string(argv[1]);
  std::string output_file = std::string(argv[2]);

  int num_nodes, num_edges;
  std::vector<int> row_offsets, col_indices, out_degrees, node_ids;
  build_graph(input_file, num_nodes, num_edges, row_offsets, col_indices,
              out_degrees, node_ids);

  // Allocate device memory
  int *d_row_offsets, *d_col_indices, *d_out_degrees;
  double *d_rank_old, *d_rank_new, *d_dangling_contrib;
  cudaMalloc(&d_row_offsets, (num_nodes + 1) * sizeof(int));
  cudaMalloc(&d_col_indices, num_edges * sizeof(int));
  cudaMalloc(&d_out_degrees, num_nodes * sizeof(int));
  cudaMalloc(&d_rank_old, num_nodes * sizeof(double));
  cudaMalloc(&d_rank_new, num_nodes * sizeof(double));
  cudaMalloc(&d_dangling_contrib, sizeof(double));

  // Copy data to device
  cudaMemcpy(d_row_offsets, row_offsets.data(), (num_nodes + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_indices, col_indices.data(), num_edges * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_out_degrees, out_degrees.data(), num_nodes * sizeof(int),
             cudaMemcpyHostToDevice);

  // Initialize rank_old
  double initial_rank = 1.0 / num_nodes;
  std::vector<double> rank_old(num_nodes, initial_rank);
  cudaMemcpy(d_rank_old, rank_old.data(), num_nodes * sizeof(double),
             cudaMemcpyHostToDevice);

  // PageRank iterations
  int max_iters = 100;
  int block_size = 256;
  int grid_size = (num_nodes + block_size - 1) / block_size;

  for (int iter = 0; iter < max_iters; ++iter) {
    // Reset dangling_contrib to zero
    double zero = 0.0;
    cudaMemcpy(d_dangling_contrib, &zero, sizeof(double),
               cudaMemcpyHostToDevice);

    // Compute dangling sum
    compute_dangling_sum<<<grid_size, block_size>>>(
        num_nodes, d_out_degrees, d_rank_old, d_dangling_contrib);
    cudaDeviceSynchronize();

    // Copy dangling_contrib back to host
    double dangling_contrib;
    cudaMemcpy(&dangling_contrib, d_dangling_contrib, sizeof(double),
               cudaMemcpyDeviceToHost);

    // PageRank kernel
    pagerank_kernel<<<grid_size, block_size>>>(
        num_nodes, d_row_offsets, d_col_indices, d_out_degrees, d_rank_old,
        d_rank_new, dangling_contrib);
    cudaDeviceSynchronize();

    // Swap rank_old and rank_new
    std::swap(d_rank_old, d_rank_new);
  }

  // Copy ranks back to host
  cudaMemcpy(rank_old.data(), d_rank_old, num_nodes * sizeof(double),
             cudaMemcpyDeviceToHost);

  // Write ranks to output file
  write_rank(node_ids, rank_old, output_file);

  // Free device memory
  cudaFree(d_row_offsets);
  cudaFree(d_col_indices);
  cudaFree(d_out_degrees);
  cudaFree(d_rank_old);
  cudaFree(d_rank_new);
  cudaFree(d_dangling_contrib);

  return 0;
}
