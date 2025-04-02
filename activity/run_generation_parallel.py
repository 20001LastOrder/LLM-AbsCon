from prompts import get_prompt
import pandas as pd
from utils import (
    extract_mermaid,
)
import json

from tqdm import tqdm
from argparse import ArgumentParser
from loguru import logger
import sys
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os
from abscon.utils import serialize_output
from dotenv import load_dotenv
from multiprocessing import Pool
import numpy as np

load_dotenv()


logger.remove()
logger.add(sys.stderr, level="INFO")


proxy = os.environ.get("OPENAI_PROXY", None)
base_url = os.environ.get("OPENAI_BASE_URL")


def process_single_result(params):
    args, sample, idx = params
    llm = get_llm(args)
    template = get_prompt(args.prompt_type)
    chain = template | llm
    text_input = sample["paragraph"]

    result = None
    while result is None:
        result_raw = chain.invoke(input={"user_input": text_input})
        result = extract_mermaid(result_raw.content)
        if result is None or len(result) == 0:
            logger.debug("result is None!")

    return result, result_raw.content, idx


def run_gpt(
    samples,
    args,
    results,
    results_raw,
    batch_size=16,
):

    num_batches = np.ceil(len(samples) / batch_size).astype(int)

    for batch_id in tqdm(range(num_batches), desc="Processing batches"):
        batch_samples = samples[batch_id * batch_size : (batch_id + 1) * batch_size]
        batch_samples = [(args, sample, i) for i, sample in enumerate(batch_samples)]

        with Pool(args.num_parallel) as pool:
            batch_all_results = list(
                tqdm(
                    pool.imap_unordered(process_single_result, batch_samples),
                    desc=f"Processing batch {batch_id}",
                    total=len(batch_samples),
                )
            )
        print([r[2] for r in batch_all_results])
        batch_all_results = sorted(batch_all_results, key=lambda r: r[2])
        print([r[2] for r in batch_all_results])
        batch_results = [r[0] for r in batch_all_results]
        batch_results_raw = [r[1] for r in batch_all_results]

        results.extend(batch_results)
        results_raw.extend(batch_results_raw)
        serialize_output(results, results_raw, args)

    return results, results_raw


def load_cache(args):
    output_path = f"{args.output_folder}/{args.llm_name}/{args.dataset}/results_{args.output_suffix}.csv"
    output_path_raw = f"{args.output_folder}/{args.llm_name}/{args.dataset}/results_{args.output_suffix}_raw.csv"

    if os.path.exists(output_path):
        results = pd.read_csv(output_path, index_col=0)
        results_raw = pd.read_csv(output_path_raw, index_col=0)
        return results["0"].to_list(), results_raw["0"].tolist()
    else:
        return [], []


def get_llm(args) -> ChatOpenAI:
    if args.llm_type == "self-hosted":
        # For some reason, directly passing the base_url to langchain results in request being blocked
        client = OpenAI(
            base_url=os.environ.get("SELF_HOSTED_LLM_URL"),
            api_key=os.environ.get("SELF_HOSTED_LLM_TOKEN"),
        )

        llm = ChatOpenAI(
            client=client.chat.completions,
            model=args.llm_name,
            temperature=args.temperature,
            request_timeout=1200,
        )
    else:
        llm = ChatOpenAI(
            base_url=base_url,
            openai_proxy=proxy,
            model=args.llm_name,
            temperature=args.temperature,
        )
    return llm


def main(args):

    with open(f"{args.input_folder}/{args.dataset}.json") as f:
        samples = json.load(f)

    results, results_raw = load_cache(args)
    num_processed = len(results)

    logger.info(f"Number of questions: {len(samples) - num_processed}")

    results, results_raw = run_gpt(
        samples[num_processed:],
        args,
        results,
        results_raw,
        batch_size=args.batch_size
    )

    serialize_output(results=results, results_raw=results_raw, args=args)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input_folder", default="data")
    parser.add_argument("--dataset", type=str, choices=["paged"], required=True)
    parser.add_argument("--output_folder", default="results")
    parser.add_argument("--output_suffix", type=str, required=True)
    parser.add_argument(
        "--prompt_type", type=str, default="fewshot", choices=["fewshot", "simple"]
    )
    parser.add_argument("--llm_type", choices=["gpt", "self-hosted"], required=True)
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_parallel", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    main(args)
