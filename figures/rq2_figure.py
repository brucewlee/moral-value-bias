import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ppq = list(range(10, 210, 10))
seed_list = ['111', '333', '555']
model_list = [
    "openai_chatgpt", "openai_chatgpt4o", "bedrock_claude3_opus",
    "bedrock_claude3_sonnet", "bedrock_claude3_haiku", "bedrock_llama3_70b_inst",
    "bedrock_llama3_8b_inst"
]
benchmark_list = ["mfq-30", "pvq-rr"]

# Create the "rq2_figures" directory if it doesn't exist
os.makedirs("rq2_figures", exist_ok=True)

def load_scores(model, benchmark, seed, ppq):
    report_file = f"../runs/persona/{benchmark}_{model}/seed{seed}_ppq{ppq}/judged_by_bedrock_claude3_haiku/report.txt"
    if not os.path.exists(report_file):
        return None
    scores = {}
    with open(report_file, "r") as f:
        content = f.read()
        if benchmark == "mfq-30":
            pattern = r"Foundation: (\w+)\nScore: ([0-9.-]+)"
        else:
            pattern = r"Value: (\w+(?:-\w+)?)\nScore: ([0-9.-]+)"
        matches = re.findall(pattern, content)
        for match in matches:
            dimension, score = match
            scores[dimension] = float(score)
    return scores

def get_dimensions(benchmark):
    if benchmark == "mfq-30":
        return ["Harm", "Fairness", "Ingroup", "Authority", "Purity"]
    else:
        return [
            "Self-Direction", "Security", "Hedonism", "Conformity",
            "Universalism", "Power", "Tradition", "Stimulation",
            "Benevolence", "Achievement"
        ]

# Calculate the middle size between MFQ and PVQ benchmarks
middle_size = (len(get_dimensions("mfq-30")) + len(get_dimensions("pvq-rr"))) / 2

def format_yticks(y, pos):
    return f"{y:.3f}"

for model in model_list:
    for benchmark in benchmark_list:
        print(f"Model: {model}, Benchmark: {benchmark}")
        fig, ax = plt.subplots(figsize=(middle_size * 0.7, 4), dpi=300)
        
        dimension_variances = {dim: [] for dim in get_dimensions(benchmark)}
        for p in ppq:
            scores_by_dimension = {dim: [] for dim in get_dimensions(benchmark)}
            for seed in seed_list:
                scores = load_scores(model, benchmark, seed, p)
                if scores is not None:
                    for dim in get_dimensions(benchmark):
                        scores_by_dimension[dim].append(scores[dim])
                        print(f"Dimension: {dim}, Score: {scores[dim]}")  # Print the scores
            for dim in get_dimensions(benchmark):
                if len(scores_by_dimension[dim]) > 0:
                    variance = np.var(scores_by_dimension[dim])
                    dimension_variances[dim].append(variance)
                else:
                    dimension_variances[dim].append(np.nan)
        
        for dim in get_dimensions(benchmark):
            ax.plot(ppq, dimension_variances[dim], linewidth=1, alpha=0.5, label=dim)
        
        avg_variances = np.nanmean(list(dimension_variances.values()), axis=0)
        ax.plot(ppq, avg_variances, linewidth=3, color="red", label="Average")
        
        ax.set_xlabel("Number of Role-Plays")
        ax.set_ylabel("Score Differences (Variance)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Format y-ticks to have 3 digits after the decimal point
        ax.yaxis.set_major_formatter(FuncFormatter(format_yticks))
        
        # Change the legend title based on the benchmark
        if benchmark == "mfq-30":
            ax.legend(title="Moral Dimensions", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            ax.legend(title="Value Dimensions", bbox_to_anchor=(1.05, 1), loc="upper left")
        
        plt.tight_layout()
        plt.savefig(f"rq2_figures/{model}_{benchmark}_variance_plot.png", dpi=300, bbox_inches='tight')
        plt.close(fig)