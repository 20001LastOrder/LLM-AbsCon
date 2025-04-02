cd ..

for num_generations in {1..20}
do 
    echo "Process {$num_generations} candidates"
    python -m scripts.generate_abscon --folder_path results/Meta-Llama-3.1-8B-Instruct --dataset_name wordnet --num_generations $num_generations
    # python -m scripts.generate_abscon --folder_path results/Meta-Llama-3.1-70B-Instruct --dataset_name wordnet --num_generations $num_generations
done