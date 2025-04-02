from openai import OpenAI
from prompts import get_prompt, get_relation
from abscon.llms import SelfHostedLLM
import pandas as pd
from output_parsers import TaxonomyOutputParser
from utils import gather_concept_groups, construct_input
from tqdm import tqdm
from argparse import ArgumentParser
from loguru import logger
import sys
from langchain_openai import ChatOpenAI
import os
from abscon.utils import serialize_output, construct_messages
from dotenv import load_dotenv
from multiprocessing import Pool

load_dotenv()


logger.remove()
logger.add(sys.stderr, level="INFO")


proxy = os.environ.get("OPENAI_PROXY", None)
base_url = os.environ.get("OPENAI_BASE_URL")


def run_single_input_openai(data):
    args, concepts, group = data

    template = get_prompt(args.dataset, args.prompt_type)
    llm = get_llm(args)
    chain = template | llm

    relation = get_relation(args.dataset)
    output_parser = TaxonomyOutputParser(
        pattern=r"```taxonomy\n((.|\n)+?)\n```", relation=relation
    )
    text_input = construct_input(concepts)

    result = None
    while result is None:
        result_raw = chain.invoke(input={"user_input": text_input})
        result = output_parser.parse(result_raw.content, group)
        if result is None or len(result) == 0:
            print("result is None!")

    return group, result, result_raw.content


def run_gpt(
    groups,
    group_concepts,
    args,
    results,
    results_raw,
):
    all_data = [(args, group_concepts[group], group) for group in groups]

    with Pool(processes=args.num_processes) as pool:
        outputs = list(
            tqdm(
                pool.imap_unordered(run_single_input_openai, all_data),
                total=len(all_data),
            )
        )

    group_result_map = {r[0]: (r[1], r[2]) for r in outputs}

    for group in groups:
        output_result = group_result_map[group]
        results.extend(output_result[0])
        results_raw.append(output_result[1])

    return results, results_raw


def load_cache(args):
    output_path = f"{args.output_folder}/{args.llm_name}/{args.dataset}/results_{args.output_suffix}.csv"
    output_path_raw = f"{args.output_folder}/{args.llm_name}/{args.dataset}/results_{args.output_suffix}_raw.csv"

    if os.path.exists(output_path):
        results = pd.read_csv(output_path, index_col=0)
        results_raw = pd.read_csv(output_path_raw, index_col=0)
        return results.to_dict(orient="records"), results_raw["0"].tolist()
    else:
        return [], []


def get_llm(args):
    if args.llm_type in "llama":
        client = OpenAI(
            base_url=os.environ.get("HOSTED_LLM_URL"),
            api_key=os.environ.get("HOSTED_LLM_TOKEN"),
        )

        llm = ChatOpenAI(
            client=client.chat.completions,
            model=args.llm_name,
            temperature=args.temperature,
        )
    elif args.llm_type == "deepseek":
        # For some reason, directly passing the base_url to langchain results in request being blocked
        client = OpenAI(
            base_url=os.environ.get("DEEPSEEK_LLM_URL"),
            api_key=os.environ.get("DEEPSEEK_LLM_TOKEN"),
        )

        llm = ChatOpenAI(
            client=client.chat.completions,
            model=args.llm_name,
            temperature=args.temperature,
        )
    else:
        llm = ChatOpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            openai_proxy=os.environ.get("OPENAI_PROXY", None),
            model=args.llm_name,
            temperature=args.temperature,
        )

    return llm


def main(args):
    relation = get_relation(args.dataset)
    output_parser = TaxonomyOutputParser(
        pattern=r"```taxonomy\n((.|\n)+?)\n```", relation=relation
    )

    test_df = pd.read_csv(f"{args.input_folder}/{args.dataset}.csv")

    results, results_raw = load_cache(args)

    processed_groups = set([str(result["group"]) for result in results])

    group_concepts = gather_concept_groups(test_df)
    group_concepts = {g: group_concepts[g] for g in groups}

    groups = set(group_concepts.keys()).difference(processed_groups)
    groups = sorted(list(groups))

    logger.info(f"Number of groups: {len(groups)}")

    results, results_raw = run_gpt(
        groups,
        group_concepts,
        args,
        results,
        results_raw,
    )

    serialize_output(results=results, results_raw=results_raw, args=args)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input_folder", default="data")
    parser.add_argument("--dataset", type=str, choices=["wordnet"], required=True)
    parser.add_argument("--output_folder", default="results")
    parser.add_argument("--output_suffix", type=str, required=True)
    parser.add_argument(
        "--llm_type", choices=["llama", "gpt", "deepseek"], required=True
    )
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument(
        "--prompt_type", choices=["simple", "fewshot"], type=str, required=True
    )

    parser.add_argument("--temperature", type=float, default=0.7)

    args = parser.parse_args()
    main(args)
