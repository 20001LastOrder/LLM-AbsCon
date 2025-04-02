from evaluation_utils import TaxonomyEvaluator
import pandas as pd
from argparse import ArgumentParser
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")


def main(args):
    evaluator = TaxonomyEvaluator(
        args.folder_path,
        args.dataset_name,
        f"{args.ground_truth_path}/{args.dataset_name}.csv",
        args.num_generations,
    )

    abscon_result_df = evaluator.generate_merged_results(
        args.num_generations, concretization_method="solver", dataset=args.dataset_name
    )
    mv_result_df = evaluator.generate_merged_results(
        args.num_generations, concretization_method="mv", dataset=args.dataset_name
    )

    mv_result_df.to_csv(
        f"{args.folder_path}/{args.dataset_name}/results_mv_{args.num_generations}.csv"
    )
    abscon_result_df.to_csv(
        f"{args.folder_path}/{args.dataset_name}/results_abscon_{args.num_generations}.csv"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("--dataset_name", choices=["ccs", "wordnet"], required=True)
    parser.add_argument("--ground_truth_path", type=str, default="data")
    parser.add_argument("--num_generations", type=int, required=True)

    args = parser.parse_args()
    main(args)
