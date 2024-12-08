#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr double DAMPING_FACTOR = 0.85;
static constexpr int MAX_ITERATIONS = 100;

struct Graph {
  int num_nodes = 0;
  int num_edges = 0;
  std::vector<int> row_offsets;
  std::vector<int> col_indicies;
  std::vector<int> out_degree;
};

Graph build_graph(const std::string &filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file");
  }

  // First pass: determine the number of nodes and edges
  // and store the edges in memory for a second pass.
  std::string line;
  std::vector<std::pair<int, int>> edges;
  int max_node_id = -1;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    std::stringstream ss(line);
    std::string token;
    std::getline(ss, token, ',');
    int from = std::stoi(token);
    std::getline(ss, token, ',');
    int to = std::stoi(token);

    edges.push_back({from, to});
    max_node_id = std::max(max_node_id, std::max(from, to));
  }
  file.close();

  Graph graph;
  graph.num_nodes = max_node_id + 1;
  graph.num_edges = static_cast<int>(edges.size());

  graph.out_degree.assign(graph.num_nodes, 0);

  for (auto &e : edges) {
    int from = e.first;
    graph.out_degree[from]++;
  }

  std::vector<std::vector<int>> adj_incoming(graph.num_nodes);
  for (auto &e : edges) {
    int from = e.first;
    int to = e.second;
    adj_incoming[to].push_back(from);
  }

  graph.row_offsets.reserve(graph.num_nodes + 1);
  graph.col_indicies.reserve(graph.num_edges);

  graph.row_offsets.push_back(0);
  for (int i = 0; i < graph.num_nodes; i++) {
    graph.row_offsets.push_back(graph.row_offsets.back() +
                                (int)adj_incoming[i].size());
    for (auto &node : adj_incoming[i]) {
      graph.col_indicies.push_back(node);
    }
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
      double sum = dangling_sum / graph.num_nodes;
      // For each incoming edge (col_indicies[j] -> i)
      for (int j = graph.row_offsets[i]; j < graph.row_offsets[i + 1]; j++) {
        int predecessor = graph.col_indicies[j];
        if (graph.out_degree[predecessor] > 0) {
          sum += rank[predecessor] / graph.out_degree[predecessor];
        }
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
    throw std::runtime_error("Could not open file for writing");
  }

  std::vector<std::pair<int, double>> rank_score;
  rank_score.reserve(rank.size());
  for (int i = 0; i < (int)rank.size(); i++) {
    rank_score.push_back({i, rank[i]});
  }

  std::sort(rank_score.begin(), rank_score.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });

  for (const auto &[node, score] : rank_score) {
    file << node << "," << score << "\n";
  }
  file.close();
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>"
              << std::endl;
    return 1;
  }

  auto start = std::chrono::high_resolution_clock::now();
  std::string input_file = argv[1];
  std::string output_file = argv[2];

  Graph graph = build_graph(input_file);
  auto preamble = std::chrono::high_resolution_clock::now();

  std::vector<double> rank = pagerank(graph);
  auto main_algo = std::chrono::high_resolution_clock::now();

  write_rank(rank, output_file);
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
