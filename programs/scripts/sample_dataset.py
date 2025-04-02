import argparse
import json
import random
import os


def get_question_type(answer: str) -> str:
    answer = answer.lower().strip()
    if answer.isnumeric():
        return "count"
    elif answer == "yes" or answer == "no":
        return "judge"
    else:
        return "query"


def main(args):
    random.seed(args.seed)

    with open(os.path.join(args.input_folder, args.question_file)) as f:
        questions = json.load(f)["questions"]
    random.shuffle(questions)

    question_types = ["count", "judge", "query"]
    selected_questions = []

    for question_type in question_types:
        count = 0
        for question in questions:
            if get_question_type(question["answer"]) == question_type:
                selected_questions.append(question)
                count += 1
            if count == args.sample_size:
                break

    count_questions = selected_questions[0 : args.sample_size]
    judge_questions = selected_questions[args.sample_size : 2 * args.sample_size]
    query_questions = selected_questions[2 * args.sample_size :]

    num_chunks = args.sample_size // args.chunk_size
    selected_questions = []

    for i in range(num_chunks):
        selected_questions.extend(
            count_questions[i * args.chunk_size : (i + 1) * args.chunk_size]
        )
        selected_questions.extend(
            judge_questions[i * args.chunk_size : (i + 1) * args.chunk_size]
        )
        selected_questions.extend(
            query_questions[i * args.chunk_size : (i + 1) * args.chunk_size]
        )

    with open(os.path.join(args.input_folder, args.output_file), "w") as f:
        json.dump({"questions": selected_questions}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default="data")
    # this is the questions from the validation set of the clevr dataset
    parser.add_argument("--question_file", default="generated_questions.json")
    parser.add_argument("--output_file", default="questions_filtered.json")
    parser.add_argument("--sample_size", default=100)
    parser.add_argument("--chunk_size", default=5)
    parser.add_argument("--seed", default=42)

    args = parser.parse_args()
    main(args)
