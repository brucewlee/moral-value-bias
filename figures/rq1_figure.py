import json
import matplotlib.pyplot as plt
import os

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

# Create the directory if it doesn't exist
os.makedirs("rq1_figures", exist_ok=True)

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
        
        # Extract the unique question numbers
        question_numbers = sorted(set(record["question_number"] for record in data))
        
        # Create a dictionary to store the count of each judge interpretation for each question number
        interpretation_counts = {qn: {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0} for qn in question_numbers}
        
        # Count the occurrences of each judge interpretation for each question number
        for record in data:
            qn = record["question_number"]
            interpretation = record["judge_interpretation"][0]  # Assuming only one interpretation per record
            try:
                interpretation_counts[qn][interpretation] += 1
            except:
                print("uncountable")
                print(interpretation)
        
        # Create the stacked bar chart
        fig, ax = plt.subplots(figsize=(len(question_numbers) * 0.6, 6))
        
        # Define pastel color scheme for colorblind-friendliness
        colors = ['#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9', '#fff2ae']
        
        bottom = [0] * len(question_numbers)
        for i, interpretation in enumerate(["A", "B", "C", "D", "E", "F"]):
            counts = [interpretation_counts[qn][interpretation] for qn in question_numbers]
            bars = ax.bar(question_numbers, counts, bottom=bottom, label=interpretation, color=colors[i], hatch='', edgecolor='black', linewidth=1)
            
            # Stripe the most popular option for each question number
            for j, bar in enumerate(bars):
                if counts[j] == max(interpretation_counts[question_numbers[j]].values()):
                    bar.set_hatch('///')
            
            bottom = [b + c for b, c in zip(bottom, counts)]
        
        # Customize the chart
        ax.set_xlabel("Question Number")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of Judge Interpretations for {benchmark} - {model}")
        ax.set_xticks(question_numbers)
        ax.set_xticklabels(question_numbers, rotation=90)
        
        # Add legend outside the chart
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        
        # Adjust layout to prevent legend from being cut off and accommodate rotated x-axis labels
        plt.subplots_adjust(right=0.8, bottom=0.2)
        
        # Save the figure
        plt.savefig(f"rq1_figures/{benchmark}_{model}.png", dpi=300, bbox_inches='tight')
        
        # Close the figure to free up memory
        plt.close(fig)