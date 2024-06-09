#!/bin/bash
#chmod +x base_runner_concurrent.sh
#base_runner_concurrent.sh

# Define the task names and model names
task_names=("mfq-30" "pvq-rr")
model_names=("cohere_commandrplus" "bedrock_llama3_8b_inst" "bedrock_llama3_70b_inst" "bedrock_claude3_haiku" "bedrock_claude3_sonnet" "bedrock_claude3_opus" "openai_chatgpt4o" "openai_chatgpt4" "openai_chatgpt")
judge_model_names=("bedrock_claude3_haiku" "openai_chatgpt" "openai_chatgpt4o" "bedrock_claude3_sonnet" "cohere_commandrplus")

# Array to store the PIDs of the background processes
pids=()

# Function to run the Python script with given task and model
run_script() {
    task_name=$1
    model_name=$2
    python base_run.py -t "$task_name" -m "$model_name" -j "${judge_model_names[@]}" &
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

# Run the script with different options concurrently
for task_name in "${task_names[@]}"
do
    for model_name in "${model_names[@]}"
    do
        run_script "$task_name" "$model_name"
    done
done

# Wait for all background processes to complete
wait