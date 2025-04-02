from prompts import get_prompt
from abscon.llms import SelfHostedLLM
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
from abscon.utils import serialize_output, construct_messages
from dotenv import load_dotenv

load_dotenv()


logger.remove()
logger.add(sys.stderr, level="INFO")


proxy = os.environ.get("OPENAI_PROXY", None)
base_url = os.environ.get("OPENAI_BASE_URL")


def run_llama(
    template,
    llm,
    samples,
    args,
    results,
    results_raw,
    serialize_frequency=1,
):
    for i, sample in enumerate(tqdm(samples)):
        text_input = sample["paragraph"]
        messages = construct_messages(template, text_input)

        result = None
        while result is None:
            response = llm(prompt="", messages=messages)
            result = extract_mermaid(response)
            if result is None or len(result) == 0:
                logger.info(f"Result is none or empty!")

        results_raw.append(response)
        results.append(result)

        if i % serialize_frequency == 0:
            serialize_output(results, results_raw, args)
    return results, results_raw


def run_gpt(
    template,
    llm,
    samples,
    args,
    results,
    results_raw,
    serialize_frequency=1,
):
    chain = template | llm

    for i, sample in enumerate(tqdm(samples)):
        text_input = sample["paragraph"]

        result = None
        while result is None:
            result_raw = chain.invoke(input={"user_input": text_input})
            result = extract_mermaid(result_raw.content)
            if result is None or len(result) == 0:
                print("result is None!")

        results.append(result)
        results_raw.append(result_raw.content)

        if i % serialize_frequency == 0:
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


def main(args):
    template = get_prompt(args.prompt_type)

    if args.llm_type == "llama":
        llm = SelfHostedLLM(
            default_model=f"{args.llm_name}",
            llm_config={"temperature": args.temperature, "max_tokens": 3000},
        )
    elif args.llm_type == "self-hosted":
        # For some reason, directly passing the base_url to langchain results in request being blocked
        client = OpenAI(
            base_url=os.environ.get("SELF_HOSTED_LLM_URL"),
            api_key=os.environ.get("SELF_HOSTED_LLM_TOKEN"),
        )

        llm = ChatOpenAI(
            client=client.chat.completions,
            model=args.llm_name,
            temperature=args.temperature,
        )
    else:
        llm = ChatOpenAI(
            base_url=base_url,
            openai_proxy=proxy,
            model=args.llm_name,
            temperature=args.temperature,
        )

    with open(f"{args.input_folder}/{args.dataset}.json") as f:
        samples = json.load(f)

    results, results_raw = load_cache(args)
    num_processed = len(results)

    logger.info(f"Number of questions: {len(samples) - num_processed}")

    if args.llm_type == "llama":
        results, results_raw = run_llama(
            template,
            llm,
            samples[num_processed:],
            args,
            results,
            results_raw,
        )
    else:
        results, results_raw = run_gpt(
            template,
            llm,
            samples[num_processed:],
            args,
            results,
            results_raw,
        )

    serialize_output(results=results, results_raw=results_raw, args=args)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input_folder", default="data")
    parser.add_argument("--dataset", type=str, choices=["paged"], required=True)
    parser.add_argument("--output_folder", default="results")
    parser.add_argument("--output_suffix", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, default="fewshot", choices=["fewshot", "simple"])
    parser.add_argument("--llm_type", choices=["llama", "gpt", "self-hosted"], required=True)
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)

    args = parser.parse_args()
    main(args)
