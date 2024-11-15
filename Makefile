
MKDIR_P = mkdir -p

.PHONY: directories

OUT_DIR = bin

all: directories pagerank-cpp pagerank-opt

debug: directories pagerank-cpp-debug pagerank-opt-debug
directories: ${OUT_DIR}

${OUT_DIR}:
	${MKDIR_P} ${OUT_DIR}


pagerank-cpp: src/pagerank.cpp
	g++ -std=c++17 -O3 -o bin/pagerank-cpp src/pagerank.cpp

pagerank-cpp-debug: src/pagerank.cpp
	g++ -std=c++17 -g -o bin/pagerank-cpp-debug src/pagerank.cpp


pagerank-opt: src/pagerank_opt.cpp
	g++ -std=c++17 -O3 -o bin/pagerank-opt src/pagerank_opt.cpp

pagerank-opt-debug: src/pagerank_opt.cpp
	g++ -std=c++17 -g -o bin/pagerank-opt-debug src/pagerank_opt.cpp

clean:
	rm -rf bin
