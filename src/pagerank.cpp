#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
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
    std::vector<int> row;
    std::stringstream ss(line);
    std::string token;
    std::getline(ss, token, ',');
    int from = std::stoi(token);
    std::getline(ss, token, ',');
    int to = std::stoi(token);
    graph[from].push_back(to);
    if (graph.find(to) == graph.end()) {
      graph[to] = {};
    }
  }
  return graph;
}

rank_t<int> pagerank(const graph_t<int> &graph) {
  rank_t<int> rank;
  rank_t<int> new_rank;
  new_rank.reserve(graph.size());

  rank.reserve(graph.size());
  for (const auto &[from, tos] : graph) {
    rank[from] = 1.0 / graph.size();
  }
  for (int i = 0; i < MAX_ITERATIONS; i++) {
    for (const auto &[from, tos] : graph) {
      double sum = 0.0;
      for (const auto &to : tos) {
        if (graph.find(to) == graph.end()) {
          continue;
        }
        sum += rank[to] / graph.at(to).size();
      }
      new_rank[from] =
          (1 - DAMPING_FACTOR) / graph.size() + DAMPING_FACTOR * sum;
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
  for (const auto &[node, score] : rank) {
    keys.push_back(node);
  }
  std::sort(keys.begin(), keys.end(),
            [&rank](int a, int b) { return rank.at(a) > rank.at(b); });
  for (const auto &key : keys) {
    file << key << "," << rank.at(key) << std::endl;
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
  graph_t<int> graph = build_graph(input_file);
  auto preamble = std::chrono::high_resolution_clock::now();
  rank_t<int> rank = pagerank(std::move(graph));
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