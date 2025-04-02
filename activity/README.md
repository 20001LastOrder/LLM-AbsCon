## Activity Diagram Generation Use Case

Using the `abscon` method to generate activity diagrams for the PAGED dataset. The original and processed datasets are available in the _data_ folder.

## Sample the WordNet Dataset dataset

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
- --dataset: paged
- --llm_type: choose from ["gpt", "llama"]
- --llm_name: choose the LLM based on whether its a GPT model of Llama model
- --temperature: use 0.7 for candidate generation and 0.01 for direct generation
- --output_folder: the folder to save the results of the LLM (default: ./results)

For example:

```bash
python run_generation.py --dataset paged --output_suffix 1 --llm_type gpt --llm_name gpt-4o-mini --temperature 0.7
```

## Generate results with abscon

Parameters:

- folder_path: path to the folder containing candidates for the LLM
- num_generation: select the number of candidate to use

For example:

```bash
python -m scripts.generate_abscon --folder_path results/gpt-4o-mini --num_generations 10
```

## Run measurements
### Check effect of candidates
```
python -m measurements.impact_of_candidates
```