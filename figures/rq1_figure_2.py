import matplotlib.pyplot as plt
import numpy as np
import os

# Function to read scores from the report files
def read_scores(file_path):
    scores = {}
    with open(file_path, 'r') as file:
        for line in file:
            if 'Score:' in line:
                value, score = line.strip().split('Score:')
                scores[value.strip()] = float(score.strip())
    return scores

# List of models and benchmarks
models = ['bedrock_claude3_haiku']
benchmarks = ['mfq-30', 'pvq-rr']

# Seeds
seeds = ['111', '333', '555']

# Create a figure and subplots for each benchmark
fig, axs = plt.subplots(1, len(benchmarks), figsize=(12, 6), subplot_kw={'projection': 'polar'})

# Iterate over benchmarks
for j, benchmark in enumerate(benchmarks):
    # Read scores for each seed and model
    for model in models:
        scores_list = []
        for seed in seeds:
            file_path = f"../runs/persona/{benchmark}_{model}/seed{seed}_ppq200/judged_by_{model}/report.txt"
            if os.path.exists(file_path):
                scores = read_scores(file_path)
                scores_list.append(scores)

        # Get the categories (foundations or values)
        categories = list(scores_list[0].keys())

        # Set the angles for each category
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)

        # Plot scores for each seed
        for k, scores in enumerate(scores_list):
            values = list(scores.values())
            values.append(values[0])  # Repeat the first value to close the plot
            axs[j].plot(angles, values, label=f"Seed {seeds[k]}")
            axs[j].fill(angles, values, alpha=0.1)

    # Set the category labels and other plot properties
    axs[j].set_xticks(angles)
    axs[j].set_xticklabels(categories)
    axs[j].set_yticklabels([])
    axs[j].set_title(f"{benchmark} - {model}")

# Add a legend to the last subplot
axs[-1].legend(loc='upper left', bbox_to_anchor=(1.2, 1.0))

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()