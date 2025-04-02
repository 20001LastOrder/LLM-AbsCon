import argparse
import json
from parser import ActivityParser
import networkx as nx
from tqdm import tqdm
from utils import partial_model_to_mermaid
import random


def diagram_malformed(diagram: nx.DiGraph):
    for node, data in diagram.nodes(data=True):
        if data["node_type"] != "decision":
            continue
        out_edges = diagram.out_edges(node, data=True)
        if len(out_edges) < 2:
            return True
        for _, _, edge_data in out_edges:
            if edge_data.get("condition", "") == "":
                return True


def main(args):
    random.seed(args.seed)

    with open(args.input_file) as f:
        dataset = json.load(f)

    parser = ActivityParser()
    valid_samples = []
    for sample in tqdm(dataset):
        graph = parser.parse(sample["mermaid_text"])
        if not diagram_malformed(graph):
            valid_samples.append(sample)

    for sample in tqdm(valid_samples):
        graph = parser.parse(sample["mermaid_text"])
        reversed_text = partial_model_to_mermaid(graph)
        reference = set(sample["mermaid_text"].split("\n"))
        reversed = set(reversed_text.split("\n"))
        assert reversed == reference

    random.shuffle(valid_samples)
    with open(args.output_file, "w") as f:
        json.dump(valid_samples[:-5], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="data/paged_raw.json")
    parser.add_argument("--output_file", default="data/paged.json")
    parser.add_argument("--seed", default=42)

    args = parser.parse_args()
    main(args)
