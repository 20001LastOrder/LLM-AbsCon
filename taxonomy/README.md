## Taxonomy Construction Use Case
Using the `abscon` method to construct taxonomies in the WordNet and CCS dataset. The original and processed datasets are available in the _data_ folder. 

## Sample the WordNet Dataset dataset
```bash
    python scripts.sample_wordnet.py --input_path data/wordnet_full.csv --output_path data/wordnet.csv
```

The filtered CCS dataset contains 75 taxonomies with fewer than 70 terms.  

## Generation
Run the following command to generate taxonomies for both datasets. It has the following configurable parameters
* dataset: choose from ["ccs", "wordnet"]
* output_suffix: choose from 1-10 for candidate generation. And give another name for the direct generation approach
* --llm_type: choose from ["gpt", "llama"]
* --llm_name: choose the LLM based on whether its a GPT model of Llama model
* --temperature: use 0.7 for candidate generation and 0.01 for direct generation

For example: 
```bash
python run_generation.py --dataset ccs --output_suffix 1 --llm_type gpt --llm_name gpt-4o-mini --temperature 0.7
```

## Generate results with abscon
Parameters:
* folder_path: path to the folder containing candidates for the LLM
* dataset_name: choose from ["wordnet", "ccs"]
* num_generation: select the number of candidate to use

For example:
```bash
python -m scripts.generate_abscon --folder_path results/gpt-4o-mini --dataset_name ccs --num_generations 10
```

## Run measurements
### Check effect of candidates
```
python -m measurements.impact_of_candidates
```