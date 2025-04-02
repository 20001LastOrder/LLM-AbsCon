from argparse import ArgumentParser
import pandas as pd
import random


def main(args):
    random.seed(args.seed)

    df = pd.read_csv(args.input_path)
    groups = df.group.unique()

    random.shuffle(groups)
    groups = groups[: args.num_samples]
    df = df[df.group.isin(groups)]

    df.child = df.child.apply(lambda s: s.replace("_", " "))
    df.parent = df.parent.apply(lambda s: s.replace("_", " "))

    df.to_csv(args.output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", default="data/wordnet_full.csv")
    parser.add_argument("--output_path", default="data/wordnet.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=100)

    args = parser.parse_args()
    main(args)
