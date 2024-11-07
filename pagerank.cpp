#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr double DAMPING_FACTOR = 0.85;

template <typename T> using graph_t = std::unordered_map<T, std::vector<T>>;
template <typename T> using rank_t = std::unordered_map<T, double>;

template <typename T> void print_graph(const graph_t<T> &graph) {
  for (const auto &[from, tos] : graph) {
    std::cout << from << ": ";
    for (const double &to : tos) {
      std::cout << to << " ";
    }
    std::cout << std::endl;
  }
}

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
  for (const auto &[from, tos] : graph) {
    rank[from] = 1.0 / graph.size();
  }
  for (int i = 0; i < 100; i++) {
    rank_t<int> new_rank;
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

    rank = new_rank;
  }
  return rank;
}

int main(int argc, char *argv[]) {
  std::string input_file = std::string(argv[1]);
  graph_t<int> graph = build_graph(input_file);
  rank_t<int> rank = pagerank(graph);

  std::vector<int> keys;
  for (const auto &[node, score] : rank) {
    keys.push_back(node);
  }
  std::sort(keys.begin(), keys.end(),
            [&rank](int a, int b) { return rank[a] > rank[b]; });

  for (const auto &key : keys) {
    std::cout << key << ": " << rank[key] << std::endl;
  }
  return 0;
}