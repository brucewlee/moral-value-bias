#!/bin/bash

# Define the task names, model names, personas per question, seeds, and judge model names
task_names=('mfq-30')
model_names=("bedrock_claude3_opus")
#model_names=("bedrock_claude3_opus" "bedrock_claude3_haiku" "bedrock_llama3_8b_inst")
#"bedrock_llama3_70b_inst" "bedrock_claude3_opus" "openai_chatgpt4o")
personas_per_question=(201)
seeds=(555)
judge_model_names=("bedrock_claude3_haiku")

# Array to store the PIDs of the background processes
pids=()

# Function to run the Python script with given parameters
run_script() {
    task_name=$1
    model_name=$2
    personas_per_question=$3
    seed=$4
    judge_model_names=$5

    python persona_run.py \
        -t "$task_name" \
        -m "$model_name" \
        -p "$personas_per_question" \
        -s "$seed" \
        -j $judge_model_names &
    pids+=($!)
}

# Function to kill all the background processes
kill_processes() {
    for pid in "${pids[@]}"
    do
        kill "$pid"
    done
}

# Trap the SIGINT signal (Ctrl+C) to kill all processes
trap kill_processes SIGINT

# Iterate over task names, model names, personas per question, and seeds concurrently
for task_name in "${task_names[@]}"
do
    for model_name in "${model_names[@]}"
    do
        for ppq in "${personas_per_question[@]}"
        do
            for seed in "${seeds[@]}"
            do
                # Join judge model names into a space-separated string
                judge_models=$(IFS=' '; echo "${judge_model_names[*]}")
                
                # Run the script with current parameters in the background
                run_script "$task_name" "$model_name" "$ppq" "$seed" "$judge_models"
            done
        done
    done
done

# Wait for all background processes to complete
wait