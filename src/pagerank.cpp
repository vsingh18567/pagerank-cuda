#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr double DAMPING_FACTOR = 0.85;
static constexpr int MAX_ITERATIONS = 100;

template <typename T> using graph_t = std::unordered_map<T, std::vector<T>>;
template <typename T> using rank_t = std::unordered_map<T, double>;

graph_t<int> build_graph(const std::string &filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file");
  }
  std::string line;
  graph_t<int> graph;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string token;
    std::getline(ss, token, ',');
    int from = std::stoi(token);
    std::getline(ss, token, ',');
    int to = std::stoi(token);

    // Add edge from -> to
    graph[from].push_back(to);

    // Ensure 'to' node exists in the graph even if it has no out-edges
    if (graph.find(to) == graph.end()) {
      graph[to] = {};
    }
  }
  file.close();
  return graph;
}

graph_t<int> build_reverse_graph(const graph_t<int> &graph) {
  graph_t<int> reverse_graph;
  // Initialize all nodes
  for (const auto &[node, tos] : graph) {
    if (reverse_graph.find(node) == reverse_graph.end()) {
      reverse_graph[node] = {};
    }
    for (const auto &to : tos) {
      reverse_graph[to].push_back(node);
      if (reverse_graph.find(to) == reverse_graph.end()) {
        reverse_graph[to] = {};
      }
    }
  }
  return reverse_graph;
}

rank_t<int> pagerank(const graph_t<int> &graph) {
  rank_t<int> rank, new_rank;
  rank.reserve(graph.size());
  new_rank.reserve(graph.size());

  // Initialize rank
  for (const auto &[node, _] : graph) {
    rank[node] = 1.0 / graph.size();
  }

  // Build reverse graph for incoming edges
  graph_t<int> reverse_graph = build_reverse_graph(graph);

  int N = (int)graph.size();

  for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
    double dangling_sum = 0.0;
    for (const auto &[node, tos] : graph) {
      if (tos.empty()) {
        dangling_sum += rank[node];
      }
    }

    for (const auto &[node, in_nodes] : reverse_graph) {
      double sum = 0.0;
      for (const auto &in_node : in_nodes) {
        int out_degree = (int)graph.at(in_node).size();
        if (out_degree > 0) {
          sum += rank[in_node] / out_degree;
        }
      }
      double dangling_contribution = dangling_sum / N;

      new_rank[node] = (1.0 - DAMPING_FACTOR) / N +
                       DAMPING_FACTOR * (sum + dangling_contribution);
    }

    std::swap(rank, new_rank);
  }

  return rank;
}

void write_rank(const rank_t<int> &rank, const std::string &filepath) {
  std::ofstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file");
  }

  std::vector<int> keys;
  keys.reserve(rank.size());
  for (const auto &[node, score] : rank) {
    keys.push_back(node);
  }

  std::sort(keys.begin(), keys.end(),
            [&rank](int a, int b) { return rank.at(a) > rank.at(b); });

  for (const auto &key : keys) {
    file << key << "," << rank.at(key) << "\n";
  }
  file.close();
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
    return 1;
  }

  auto start = std::chrono::high_resolution_clock::now();
  std::string input_file = argv[1];
  std::string output_file = argv[2];

  graph_t<int> graph = build_graph(input_file);
  auto preamble = std::chrono::high_resolution_clock::now();

  rank_t<int> final_rank = pagerank(graph);
  auto main_algo = std::chrono::high_resolution_clock::now();

  write_rank(final_rank, output_file);
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
