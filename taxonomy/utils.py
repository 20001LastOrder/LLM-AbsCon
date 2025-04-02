import networkx as nx
import pandas as pd
from tqdm import tqdm
import numpy as np


def gather_concept_groups(df) -> dict[str, list[str]]:
    groups = df.group.unique()
    result = {}

    for group in groups:
        group_df = df[df.group == group]
        concepts = set(group_df.parent).union(set(group_df.child))
        result[str(group)] = concepts
    return result


def construct_input(concepts) -> str:
    return f"Concepts: {', '.join(concepts)}"


def dataframe_to_ancestor_graph(df, verbose=False):
    forest = []

    if not verbose:
        tqdm_func = lambda a: a
    else:
        tqdm_func = lambda a: tqdm(a)

    for group in tqdm_func(list(set(df.group))):
        forest.append(group_to_ancestor_graph(df, group))
    return pd.concat(forest, ignore_index=True)


def group_to_ancestor_graph(df, g):
    df_tree = df[df.group == g]
    graph = nx.DiGraph()
    parents = df_tree["parent"].apply(lambda a: str(a).lower()).tolist()
    children = df_tree["child"].apply(lambda a: str(a).lower()).tolist()
    nodes = set(parents + children)
    for node in nodes:
        graph.add_node(node)
    for i in range(len(parents)):
        graph.add_edge(parents[i], children[i])

    T = convert_to_ancestor_graph(graph)

    df = nx.to_pandas_edgelist(T)
    df["group"] = g
    df.columns = ["parent", "child", "group"]
    df["compare"] = df["parent"] + df["child"] + df["group"].astype(str)

    return df


def df_to_graph(df):
    graph = nx.DiGraph()
    parents = df["parent"].apply(lambda a: str(a).lower()).tolist()
    children = df["child"].apply(lambda a: str(a).lower()).tolist()
    nodes = set(parents + children)
    node_map = {}

    for node_id, label in enumerate(nodes):
        graph.add_node(node_id, label=label)
        node_map[label] = node_id

    for i in range(len(parents)):
        graph.add_edge(node_map[parents[i]], node_map[children[i]])

    return graph


def graph_to_df(graph):
    relations = []
    group = graph.graph["group"]

    for parent, child in graph.edges:
        parent_text = graph.nodes[parent]["label"]
        child_text = graph.nodes[child]["label"]

        relations.append({"child": child_text, "parent": parent_text, "group": group})

    return pd.DataFrame(relations)


def convert_to_ancestor_graph(G):
    """Converts a (parent) tree to a graph with edges for all ancestor relations in the tree."""
    G_anc = nx.DiGraph()
    for node in G.nodes():
        for anc in nx.ancestors(G, node):
            G_anc.add_edge(anc, node)
    return G_anc


def evaluate_group_consistency(
    df, num_groups, one_parent_constraint=True, return_value="avg", df_actual=None
):
    groups = (
        sorted(list(set(df.group)))
        if df_actual is None
        else sorted(list(set(df_actual.group)))
    )
    consistency = []
    num_violations_counts = []

    for i, group in enumerate(groups):
        df_group = df[df.group == group]
        graph = df_to_graph(df_group)

        if len(graph.nodes) == 0:
            consistency.append(True)
            num_violations_counts.append(0)
            continue
        consistency_results = evaluate_consistency(graph)

        consistent = True
        num_violations = 0
        for name, result in consistency_results.items():
            if not one_parent_constraint and name == "parent":
                continue
            consistent = consistent and result["satisfied"]
            num_violations += result["num_violations"]

        num_violations_counts.append(num_violations)
        consistency.append(consistent)

    if return_value == "avg":
        return sum(consistency) / num_groups
    else:
        return {
            "consistency": consistency,
            "num_violations": num_violations_counts
        }


def evaluate_consistency(graph: nx.DiGraph) -> dict[str, dict]:
    # check the number of cycles
    num_cycles = 0
    for _ in nx.simple_cycles(graph):
        num_cycles += 1
        break

    # check the number of roots
    num_roots = 0
    for node in graph.nodes:
        if graph.in_degree(node) == 0:
            num_roots += 1

    # check the number of skip connections:
    num_skips = 0
    for src in graph.nodes:
        for tgt in graph.nodes:
            if not graph.has_edge(src, tgt):
                continue
            if src == tgt:
                continue
            for path in nx.all_simple_paths(graph, src, tgt):
                if len(path) <= 2:
                    continue
                num_skips += 1
                break
        if num_skips >= 1:
            break

    # check the number of nodes with more than one parent
    num_multi_parent = 0
    for node in graph.nodes:
        if graph.in_degree(node) > 1:
            num_multi_parent += 1

    return {
        "cycles": {"satisfied": num_cycles == 0, "num_violations": num_cycles},
        "roots": {"satisfied": num_roots == 1, "num_violations": abs(num_roots - 1)},
        "skips": {"satisfied": num_skips == 0, "num_violations": num_skips},
        "parent": {"satisfied": num_multi_parent == 0, "num_violations": num_multi_parent},
    }


def evaluate_groups(df_actual, df_pred, verbose=False, return_value="avg"):
    recall = []
    precision = []
    f1 = []

    if not verbose:
        tqdm_func = lambda a: a
    else:
        tqdm_func = lambda a: tqdm(a)
    for group in tqdm_func(sorted(list(set(df_actual.group)))):
        group_actual = df_actual[df_actual.group == group]
        group_pred = df_pred[df_pred.group == group]
        group_common = pd.merge(group_actual, group_pred, on=["compare"], how="inner")

        group_recall = (
            len(group_common) / len(group_actual) if len(group_actual) > 0 else 0
        )
        group_precision = (
            len(group_common) / len(group_pred) if len(group_pred) > 0 else 0
        )

        if group_recall + group_precision == 0:
            group_f1 = 0
        else:
            group_f1 = (
                2 * (group_precision * group_recall) / (group_precision + group_recall)
            )

        recall.append(group_recall)
        precision.append(group_precision)
        f1.append(group_f1)

    if return_value == "avg":
        return np.mean(recall), np.mean(precision), np.mean(f1)
    else:
        return recall, precision, f1
