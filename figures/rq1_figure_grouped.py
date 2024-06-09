import json
import matplotlib.pyplot as plt
import os
import numpy as np

model_list = [
    "openai_chatgpt",
    "openai_chatgpt4o",
    "bedrock_claude3_opus",
    "bedrock_claude3_sonnet",
    "bedrock_claude3_haiku",
    "bedrock_llama3_70b_inst",
    "bedrock_llama3_8b_inst"
]
benchmark_list = [
    "mfq-30", "pvq-rr"
]

foundations = {
    'Harm': [1, 7, 12, 17, 23, 28],
    'Fairness': [2, 8, 13, 18, 24, 29],
    'Ingroup': [3, 9, 14, 19, 25, 30],
    'Authority': [4, 10, 15, 20, 26, 31],
    'Purity': [5, 11, 16, 21, 27, 32]
}

values_mapping_10 = {
    "Self-Direction": [1,23,39,16,30,56],
    "Stimulation": [10,28,43],
    "Hedonism": [3,36,46],
    "Achievement": [17,32,48],
    "Power": [6,29,41,12,20,44],
    "Security": [13,26,53,2,35,50],
    "Conformity": [15,31,42,4,22,51],
    "Tradition": [18,33,40,7,38,54],
    "Benevolence": [11,25,47,19,27,55],
    "Universalism": [8,21,45,5,37,52,14,34,57]
}

# Create the directory if it doesn't exist
os.makedirs("rq1_figures_grouped", exist_ok=True)

for benchmark in benchmark_list:
    for model in model_list:
        file_path = f"../runs/persona/{benchmark}_{model}/seed111_ppq200/judged_by_bedrock_claude3_haiku/records.jsonl"

        try:
            # Read the JSON data from the file
            with open(file_path, "r") as file:
                data = [json.loads(line) for line in file]
        except:
            print(f"{file_path} doesn't exist yet")
            continue
        
        # Determine the grouping based on the benchmark
        grouping = foundations if benchmark == "mfq-30" else values_mapping_10
        
        # Create a dictionary to store the count of each judge interpretation for each group
        interpretation_counts = {group: {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0} for group in grouping}
        
        # Count the occurrences of each judge interpretation for each group
        for record in data:
            qn = record["question_number"]
            interpretation = record["judge_interpretation"][0]  # Assuming only one interpretation per record
            for group, questions in grouping.items():
                if qn in questions:
                    try:
                        interpretation_counts[group][interpretation] += 1
                    except:
                        print("uncountable")
                        print(interpretation)
        
        # Calculate the percentage for each group
        for group in grouping:
            total_count = sum(interpretation_counts[group].values())
            for interpretation in interpretation_counts[group]:
                interpretation_counts[group][interpretation] = (interpretation_counts[group][interpretation] / total_count) * 100
        
        # Create the stacked bar chart with adjusted figure size
        fig, ax = plt.subplots(figsize=(len(grouping) * 1.8, 4))
        
        # Define the bipolar color gradient
        blue_gradient = ['#deebf7', '#c6dbef', '#9ecae1']
        red_gradient = ['#fee0d2', '#fcbba1', '#fc9272']
        
        bottom = [0] * len(grouping)
        for i, interpretation in enumerate(["A", "B", "C", "D", "E", "F"]):
            percentages = [interpretation_counts[group][interpretation] for group in grouping]
            color = blue_gradient[i] if i < 3 else red_gradient[i-3]
            bars = ax.bar(grouping.keys(), percentages, bottom=bottom, label=interpretation, color=color, hatch='', edgecolor='black', linewidth=1)
            
            # Stripe the most popular option for each group
            for j, bar in enumerate(bars):
                if percentages[j] == max(interpretation_counts[list(grouping.keys())[j]].values()):
                    bar.set_hatch('///')
            
            bottom = [b + p for b, p in zip(bottom, percentages)]
        
        # Customize the chart
        ax.set_xlabel("Group")
        ax.set_ylabel("Percentage")
        ax.set_title(f"Distribution of Judge Interpretations for {benchmark} - {model}")
        ax.set_xticklabels(grouping.keys(), rotation=45, ha='right')
        ax.set_ylim(0, 100)
        
        # Add legend outside the chart
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        
        # Adjust layout to prevent legend from being cut off and accommodate rotated x-axis labels
        plt.subplots_adjust(right=0.8, bottom=0.3)
        
        # Save the figure
        plt.savefig(f"rq1_figures_grouped/{benchmark}_{model}_grouped_percentage.png", dpi=300, bbox_inches='tight')
        
        # Close the figure to free up memory
        plt.close(fig)