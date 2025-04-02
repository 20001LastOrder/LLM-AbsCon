from evaluation_utils import ClevrEvaluator
import pandas as pd
from argparse import ArgumentParser
from loguru import logger
import sys
from sentence_transformers import SentenceTransformer
import random
import numpy as np

logger.remove()
logger.add(sys.stderr, level="INFO")


def main(args):
    evaluator = ClevrEvaluator(
        args.folder_path,
        args.dataset_name,
        data_folder=args.ground_truth_path,
        scene_file=args.scene_file,
        seed=args.seed
    )

    encoder = SentenceTransformer(args.encoder, local_files_only=True)

    for num_candidates in range(args.num_candidates_start, args.num_candidates_end + 1):
        logger.info(f"Processing {num_candidates} candidates...")
        mv_results = evaluator.combine_solutions(
            num_candidates, encoder, concretization_method="mv", verbose=True
        )
        abscon_results = evaluator.combine_solutions(
            num_candidates, encoder, concretization_method="solver", verbose=True
        )

        pd.DataFrame(mv_results).to_csv(
            f"{args.folder_path}/{args.dataset_name}/results_mv_{num_candidates}.csv"
        )
        pd.DataFrame(abscon_results).to_csv(
            f"{args.folder_path}/{args.dataset_name}/results_abscon_{num_candidates}.csv"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("--dataset_name", choices=["clevr"], default="clevr")
    parser.add_argument("--ground_truth_path", type=str, default="data")
    parser.add_argument("--scene_file", type=str, default="scenes.json")
    parser.add_argument(
        "--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--num_candidates_start",
        help="Starting number of candidates to abstract, this is included",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--num_candidates_end",
        help="Ending number of candidates to abstract (inclusive). If -1 then it will be set the same number as num_candidates_start",
        type=int,
        default=False,
    )
    parser.add_argument(
        "--seed",
        help="The random seed to control randomness in the approximation algorithms",
        type=int,
        default=42
    )

    args = parser.parse_args()

    if args.num_candidates_end == -1:
        args.num_candidates_end = args.num_candidates_start

    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)
