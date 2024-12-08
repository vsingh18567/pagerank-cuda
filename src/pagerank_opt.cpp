/*
1. Uses CSR format to store the graph.
2. Handles dangling nodes.
*/
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr double DAMPING_FACTOR = 0.85;
static constexpr int MAX_ITERATIONS = 100;

struct Graph {
  int num_nodes;
  int num_edges;
  std::vector<int> row_offsets;
  std::vector<int> col_indicies;
  std::vector<int> out_degree;
};

Graph build_graph(const std::string &filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file");
  }
  std::string line;
  Graph graph;
  std::vector<std::vector<int>> adj_incoming;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string token;
    std::getline(ss, token, ',');
    int from = std::stoi(token);
    std::getline(ss, token, ',');
    int to = std::stoi(token);
    auto max_node = std::max(from, to);
    if (max_node >= graph.num_nodes) {
      graph.num_nodes = max_node + 1;
    }
    // std::cout << "from: " << from << " to: " << to << std::endl;
    graph.num_edges++;
    adj_incoming.resize(graph.num_nodes);
    adj_incoming[to].push_back(from);
  }
  file.close();
  graph.row_offsets.reserve(graph.num_nodes + 1);
  graph.col_indicies.reserve(graph.num_edges);
  graph.out_degree.reserve(graph.num_nodes);

  graph.row_offsets.push_back(0);

  for (int i = 0; i < graph.num_nodes; i++) {
    graph.row_offsets.push_back(graph.row_offsets.back() +
                                adj_incoming[i].size());
    for (const auto &node : adj_incoming[i]) {
      graph.col_indicies.push_back(node);
    }
    graph.out_degree.push_back(adj_incoming[i].size());
  }
  return graph;
}

std::vector<double> pagerank(const Graph &graph) {
  std::vector<double> rank_new(graph.num_nodes, 0.0);
  std::vector<double> rank(graph.num_nodes, 1.0 / graph.num_nodes);
  for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
    double dangling_sum = 0.0;
    for (int i = 0; i < graph.num_nodes; i++) {
      if (graph.out_degree[i] == 0) {
        dangling_sum += rank[i];
      }
    }
    for (int i = 0; i < graph.num_nodes; i++) {
      double sum = (dangling_sum / graph.num_nodes);
      for (int j = graph.row_offsets[i]; j < graph.row_offsets[i + 1]; j++) {
        if (graph.out_degree[graph.col_indicies[j]] == 0) {
          continue;
        }
        sum += rank[graph.col_indicies[j]] /
               graph.out_degree[graph.col_indicies[j]];
      }
      rank_new[i] =
          (1.0 - DAMPING_FACTOR) / graph.num_nodes + DAMPING_FACTOR * sum;
    }
    std::swap(rank, rank_new);
  }
  return rank;
}

void write_rank(const std::vector<double> &rank, const std::string &filepath) {
  std::ofstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file");
  }

  std::vector<std::pair<int, double>> rank_score;
  for (int i = 0; i < rank.size(); i++) {
    rank_score.push_back({i, rank[i]});
  }
  std::sort(rank_score.begin(), rank_score.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });
  for (const auto &[node, score] : rank_score) {
    file << node << "," << score << std::endl;
  }
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>"
              << std::endl;
    return 1;
  }
  auto start = std::chrono::high_resolution_clock::now();
  std::string input_file = std::string(argv[1]);
  std::string output_file = std::string(argv[2]);
  Graph graph = build_graph(input_file);
  auto preamble = std::chrono::high_resolution_clock::now();
  std::vector<double> rank = pagerank(std::move(graph));
  auto main_algo = std::chrono::high_resolution_clock::now();
  write_rank(std::move(rank), output_file);
  auto write_output = std::chrono::high_resolution_clock::now();
  std::cout << "Preamble: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(preamble -
                                                                     start)
                   .count()
            << "ms\n";
  std::cout << "Main Algorithm: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(main_algo -
                                                                     preamble)
                   .count()
            << "ms\n";
  std::cout << "Write Output: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   write_output - main_algo)
                   .count()
            << "ms\n";
  std::cout << "Total: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   write_output - start)
                   .count()
            << "ms\n";
  return 0;
}
