## Program Induction Use Case

Using the `abscon` method to induct programs for the Clevr dataset. The original and processed datasets are available in the _data_ folder.

## Sample the dataset

```bash
    python -m scripts.sample_dataset
```

- The arguments and default values are available in this file

## Generation

Set the following two environment variables for OpenAI:

- OPENAI_BASE_URL
- OPENAI_PROXY (if necessary)

Run the following command to generate programs from the sampled questions. It has the following required parameters

- output_suffix: choose from 1-10 for candidate generation. And give another name for the direct generation approach
- --dataset: clevr
- --llm_type: choose from ["gpt", "llama"]
- --llm_name: choose the LLM based on whether its a GPT model of Llama model
- --temperature: use 0.7 for candidate generation and 0.01 for direct generation

For example:

```bash
python run_generation.py --dataset clevr --output_suffix 1 --llm_type gpt --llm_name gpt-4o-mini --temperature 0.7
```

## Generate results with abscon

Parameters:

- folder_path: path to the folder containing candidates for the LLM
- num_candidates_start: select the number of candidates to begin producing
- num_candidates_end: optional. Select the number of candidates when ending merging (inclusive). If no value is provided, then it will be set the same as num_candidates_start

For example, to generate a concretized model by abstracting 10 candidates from gpt-4o-mini:

```bash
python -m scripts.generate_abscon --folder_path results/gpt-4o-mini --num_candidates_start 10
```

## Run measurements

### Check effect of candidates
```
python -m measurements.impact_of_candidates
```

### Effect of randomness
* When there are no possible concretized model or if there are multiple possible best solutions, it is possible that a different solution may be obtained for different runs. A seed is set to preserve determinism

* During abstraction and concretized, a timeout is set to return the best current solution within a certain number of time. Depending on the computation speed, a different result may be obtained within the timeout limit. However, such cases should be rare and do not have significant impact on the results. 