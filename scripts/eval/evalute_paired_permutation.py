# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Get summary.csv with score and null predictions amount.

Running
```
python evaluate.py \
    --data_dir /path/to/your/prediction_jsonl_folder \
    --benchmark synthetic
```
"""

import argparse
import os
import re
import numpy as np
from scipy.stats import permutation_test

import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

import importlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml
from nemo.collections.asr.parts.utils.manifest_utils import (read_manifest,
                                                             write_manifest)
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--benchmark", type=str, default="synthetic", help="Options: [synthetic]"
)
parser.add_argument("--seq-len", type=int, default=0)
parser.add_argument(
    "--verbose", type=int, default=0, help="how many lines you want to display."
)
parser.add_argument("--dir-one", type=str, default="", help="dir of the first model")
parser.add_argument(
    "--model-one", type=str, default="", help="the first model for the paired permutation test."
)

parser.add_argument("--dir-two", type=str, default="", help="dir of the second model")
parser.add_argument(
    "--model-two", type=str, default="", help="the second model for the paired permutation test."
)
args = parser.parse_args()


def postprocess_pred(predict_str: str, task_config: dict):

    predict_str = predict_str.strip()

    # Remove all non-printable characters
    np_pattern = re.compile(r"[\x00-\x1f]")
    predict_str = np_pattern.sub("\n", predict_str).strip()

    return predict_str


def get_pred_and_ref(
    predictions_file: str,
    task_config: dict,
    input_field: str = "input",
    references_field: str = "outputs",
    prediction_field: str = "pred",
    metadata_field: str = "others",
):
    lines = read_manifest(predictions_file)

    inputs = []
    predicts = []
    references = []
    indices = []

    for line in tqdm(lines):
        input = line[input_field]
        predict = line[prediction_field]
        predict = postprocess_pred(predict, task_config)
        reference = line.get(references_field, [line.get("output", "")])
        index = line[metadata_field].get("id", line["index"])

        inputs.append(input)
        predicts.append(predict)
        references.append(reference)
        indices.append(index)

    return inputs, predicts, references, indices


def run_evaluation_per_task(task_config: dict, predictions_file: str, verbose: int = 0):
    inputs, predicts, references, indices = get_pred_and_ref(
        predictions_file=predictions_file,
        task_config=task_config,
    )

    task_nulls = f"{sum([len(x)==0 for x in predicts])}/{len(predicts)}"

    if len(references) > 0 and references[0][0] is not None:
        task_scores = task_config["metric_fn"](predicts, references, reduce=False)
    else:
        raise ValueError(f"something is wrong....")
        # task_score = [0 for _ in indices]

    if verbose != 0:
        print("=" * 40)
        for i, (input, reference, predict) in enumerate(
            zip(inputs, references, predicts)
        ):
            print(f"Input     : {input}")
            print(f"Reference : {reference}")
            print(f"Prediction: {predict}")
            print("=" * 40)
            if i > verbose:
                break

    return task_scores, task_nulls, predicts, indices


def write_evaluation(pvalues_greater: dict, pvalues_less: dict):

    for i, (d, alt) in enumerate(zip((pvalues_greater, pvalues_less), ("greater", "less"))):
        tasks = list(d.keys())
        pvalues = [d[task] for task in tasks]

        if i == 0:
            dfs = [
                ["tasks"] + tasks,
            ]
        dfs.append([f"pvalues-{alt}"] + pvalues)

    path = f"/data/jeff/RULER/scripts/benchmark_root/llama3.1-8b-chat/synthetic/{args.seq_len}/paired-perm"
    os.makedirs(path, exist_ok=True)

    output_file = os.path.join(path, f"{args.model_one}--{args.model_two}.csv")

    df = pd.DataFrame(dfs)
    df.to_csv(output_file, index=False)
    print("\n=============================================\n")
    print(f"{df}")
    print("\n=============================================\n")
    # print(f"\nSaved eval results to {output_file}")


def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def main():
    curr_folder = os.path.dirname(os.path.abspath(__file__))

    try:
        module = importlib.import_module(f"{args.benchmark}.constants")
    except ImportError:
        print(f"Module eval.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    TASKS = tasks_customized
    for _, config in TASKS.items():
        config.update(tasks_base[config["task"]])

    print(f"Total tasks: {list(TASKS.keys())}")

    # Aggregate all prediction files
    # aggregate_chunk(args.data_dir)

    # Get scores and nulls
    model_one = {"all": [], "part": []}
    model_two = {"all": [], "part": []}

    pvalues_less = {}
    pvalues_greater = {}
    for alternative, pvalues in zip(["greater", "less"], [pvalues_greater, pvalues_less]):
        for task, config in TASKS.items():
            if "cwe" in task:
                continue

            for fname, dr, d in [
                (args.model_one, args.dir_one, model_one),
                (args.model_two, args.dir_two, model_two)
            ]:


                if fname != "":
                    fname = "-" + fname
                path = os.path.join(dr, f"{task}{fname}.jsonl")
                if not os.path.exists(path):
                    print(f"Prediction file {path} is not found.")
                    continue

                print(f"Evaluate task {task}...")

                task_scores, task_nulls, predicts, indices = run_evaluation_per_task(
                    predictions_file=path,
                    task_config=config,
                )

                d[task] = task_scores
                if "qa" not in task:
                    d["all"] += task_scores
                else:
                    d["part"] += task_scores

            # print(f"doing paired permutation test")
            print(f"delta: {np.mean(model_one[task])} baseline: {np.mean(model_two[task])}")
            res = permutation_test(
                (model_one[task], model_two[task]),
                statistic,
                permutation_type="samples",
                vectorized=True,
                n_resamples=9999,
                alternative=alternative
            )
            pvalues[task] = res.pvalue
            print(f"{res.pvalue=}")

        res = permutation_test(
            (model_one["all"], model_two["all"]),
            statistic,
            permutation_type="samples",
            vectorized=True,
            n_resamples=9999,
            alternative=alternative
        )

        pvalues["all"] = res.pvalue
        # print(f"all: {res.pvalue=}")
        res = permutation_test(
            (model_one["part"], model_two["part"]),
            statistic,
            permutation_type="samples",
            vectorized=True,
            n_resamples=9999,
            alternative=alternative
        )
        # print(f"part: {res.pvalue=}")
        pvalues["part"] = res.pvalue

        
    write_evaluation(pvalues_greater, pvalues_less)


if __name__ == "__main__":
    main()
