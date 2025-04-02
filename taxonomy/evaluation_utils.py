import pandas as pd
import os
from utils import (
    dataframe_to_ancestor_graph,
    evaluate_groups,
    df_to_graph,
    graph_to_df,
    evaluate_group_consistency,
)
import numpy as np
from tqdm import tqdm
from abscon.abstraction import TaxonomyAbstractor
from abscon.concretization import TaxonomyConcretizer, MajorityVotingConcretizer


class TaxonomyEvaluator:
    def __init__(
        self,
        folder_path,
        dataset_name,
        ground_truth_path,
        num_generations=5,
        evaluate_greedy=False,
        filenames = []
    ):
        data_dir = os.path.join(folder_path, dataset_name)

        if filenames == []:
            if not evaluate_greedy:
                filenames = [
                    f"{data_dir}/results_{i}.csv" for i in range(1, num_generations + 1)
                ]
            else:
                filenames = [f"{data_dir}/results_greedy.csv"]

        dfs = []
    
        for filename in filenames:
            df = pd.read_csv(filename)
            df["child"] = df["child"].apply(lambda x: str(x).replace(" ", "_"))
            df["parent"] = df["parent"].apply(lambda x: str(x).replace(" ", "_"))
            dfs.append(df)

        # get ground truth
        groups = set()
        for df in dfs:
            groups = groups.union(set(df.group.unique()))

        df_actual = pd.read_csv(ground_truth_path)
        # df_actual=df_actual[df_actual['type'] == 'test']
        df_actual = df_actual[df_actual.group.isin(groups)]
        df_actual["child"] = df_actual["child"].apply(lambda x: x.replace(" ", "_"))
        df_actual["parent"] = df_actual["parent"].apply(lambda x: x.replace(" ", "_"))

        actual_graphs = {}
        for group in df_actual.group:
            actual_graphs[group] = df_to_graph(df_actual[df_actual.group == group])

        group_to_graphs = {}

        for group in groups:
            graphs = []
            for df in dfs:
                graphs.append(df_to_graph(df[df.group == group]))
            group_to_graphs[group] = graphs

        self.dfs = dfs
        self.groups = groups
        self.df_actual = dataframe_to_ancestor_graph(df_actual)
        self.actual_graphs = actual_graphs
        self.group_to_graphs = group_to_graphs

    def generate_merged_results(
        self,
        num_samples,
        concretization_method="solver",
        start_sample=0,
        dataset="wordnet",
        verbose=False,
    ):
        one_parent_constraint = dataset == "wordnet"
        group_to_partial_graph = {}
        for group, graphs in self.group_to_graphs.items():
            node_labels = sorted(
                [
                    data["label"]
                    for _, data in self.actual_graphs[group].nodes(data=True)
                ]
            )
            abstractor = TaxonomyAbstractor(nodes=node_labels)
            for graph in graphs[start_sample : start_sample + num_samples]:
                abstractor.add_concrete_model(graph)
            group_to_partial_graph[group] = abstractor.get_partial_model()

        taxonomies = []
        if not verbose:
            tqdm_func = lambda a: a
        else:
            tqdm_func = lambda a: tqdm(a)

        for group in tqdm_func(list(group_to_partial_graph.keys())):

            partial_model = group_to_partial_graph[group]

            if concretization_method == "solver":
                concretizer = TaxonomyConcretizer()
            else:
                concretizer = MajorityVotingConcretizer()

            taxonomy = concretizer.concretize(
                partial_model, one_parent_constraint=one_parent_constraint
            )
            taxonomy.graph["group"] = group
            taxonomies.append(taxonomy)

        dfs = []
        for taxonomy in taxonomies:
            dfs.append(graph_to_df(taxonomy))
        final_df = pd.concat(dfs)

        return final_df

    def evaluate_abstraction(
        self,
        num_samples,
        concretization_method="solver",
        dataset="wordnet",
        verbose=False,
    ):
        one_parent_constraint = dataset == "wordnet"

        group_to_partial_graph = {}
        for group, graphs in self.group_to_graphs.items():
            node_labels = sorted(
                [
                    data["label"]
                    for _, data in self.actual_graphs[group].nodes(data=True)
                ]
            )
            abstractor = TaxonomyAbstractor(nodes=node_labels)
            for graph in graphs[:num_samples]:
                abstractor.add_concrete_model(graph)
            group_to_partial_graph[group] = abstractor.get_partial_model()

        taxonomies = []
        if not verbose:
            tqdm_func = lambda a: a
        else:
            tqdm_func = lambda a: tqdm(a)

        for group in tqdm_func(list(group_to_partial_graph.keys())):

            partial_model = group_to_partial_graph[group]

            if concretization_method == "solver":
                concretizer = TaxonomyConcretizer()
            else:
                concretizer = MajorityVotingConcretizer()

            taxonomy = concretizer.concretize(
                partial_model, one_parent_constraint=one_parent_constraint
            )
            taxonomy.graph["group"] = group
            taxonomies.append(taxonomy)

        dfs = []
        for taxonomy in taxonomies:
            dfs.append(graph_to_df(taxonomy))
        final_df = pd.concat(dfs)

        consistency = evaluate_group_consistency(
            final_df, len(group_to_partial_graph), one_parent_constraint
        )

        df_actual = dataframe_to_ancestor_graph(self.df_actual)
        final_df = dataframe_to_ancestor_graph(final_df)

        recall, precision, f1 = evaluate_groups(df_actual, final_df)

        return {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "consistency": consistency,
        }

    def evaluate_individual(self, runs, dataset, aggregator=np.mean):
        results = []
        for run in range(runs):
            final_df = self.generate_merged_results(
                num_samples=1,
                concretization_method="mv",
                start_sample=run,
                dataset=dataset,
            )
            metrics = self.evaluate_taxonomies(final_df, dataset, return_value="all")
            results.append(metrics)
        metrics = ["f1", "recall", "precision"]

        metric_result = {}
        for metric in metrics:
            metric_values = [
                aggregator([results[run][metric][group_id] for run in range(runs)])
                for group_id in range(len(results[0][metric]))
            ]
            metric_result[metric] = np.mean(metric_values)
        return metric_result

    def evaluate_taxonomies(self, final_df, dataset, return_value="avg"):
        one_parent_constraint = dataset == "wordnet"
        num_groups = len(self.group_to_graphs)
        df_actual = dataframe_to_ancestor_graph(self.df_actual)

        if return_value == "avg":
            consistency = evaluate_group_consistency(
                final_df, num_groups, one_parent_constraint
            )
        else:
            consistency = evaluate_group_consistency(
                final_df, num_groups, one_parent_constraint, return_value, df_actual
            )

        final_df = dataframe_to_ancestor_graph(final_df)

        recall, precision, f1 = evaluate_groups(
            df_actual, final_df, return_value=return_value
        )

        return {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "consistency": consistency,
        }
