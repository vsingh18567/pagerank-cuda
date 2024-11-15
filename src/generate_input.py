import random
import argparse
from tqdm import tqdm


def generate_input(num_nodes: int, num_edges: int, filename: str):
    with open(filename, "w") as f:
        for i in tqdm(range(num_edges)):
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            while u == v:
                v = random.randint(0, num_nodes - 1)
            f.write(f"{u},{v}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_nodes", type=int)
    parser.add_argument("num_edges", type=int)
    parser.add_argument("filename", type=str)
    args = parser.parse_args()
    generate_input(args.num_nodes, args.num_edges, args.filename)


if __name__ == "__main__":
    main()
