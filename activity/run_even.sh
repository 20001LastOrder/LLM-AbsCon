for num_generations in 12 14 16 18 20
do 
    echo "Process generation {$num_generations}"
    python run_generation_parallel.py --dataset paged --output_suffix ${num_generations} --llm_type self-hosted --llm_name <name> --temperature 0.7 --num_parallel 64 --batch_size 128 --prompt_type simple 
done

