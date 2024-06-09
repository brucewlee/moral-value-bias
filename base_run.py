from api_keys import set_api_keys
set_api_keys()
import os, logging, sys
logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.CRITICAL)
from nutcracker.data import Task, Pile
from nutcracker.runs import Schema
from nutcracker.evaluator import AutoEvaluator, mfq_30_generate_report, pvq_rr_generate_report
import argparse
from utils import model_selector



def main_runner(task_name, model_name, judge_model_names):
    task = Task.load_from_db(
        task_name = task_name, 
        db_directory ='../nutcracker-db/db'
    )

    # sample for demo
    #task.sample(5, in_place=True)

    # running this experiment updates each instance's model_response property in truthfulqa data object with ChatGPT responses
    experiment = Schema(
        model = model_selector(model_name),
        data = task
        )
    experiment.run()

    for judge_model_name in judge_model_names:
        # running this evaluation updates each instance's response_correct property in truthfulqa data object with evaluations
        evaluation = AutoEvaluator(
            model = model_selector(judge_model_name),
            data = task
            )
        evaluation.run()

        # Create a new directory based on the command-line arguments
        output_dir = f"runs/base/{task_name}_{model_name}/judged_by_{judge_model_name}"
        os.makedirs(output_dir, exist_ok=True)

        task.save_records(f"{output_dir}/records.jsonl", keys = ["question_number", "centerpiece", "options", "persona", "judge_interpretation", "user_prompt", "model_response"])

        if task_name == "mfq-30":
            mfq_30_generate_report(task, f'{output_dir}/report.txt')
        elif task_name == "pvq-rr":
            pvq_rr_generate_report(task, f'{output_dir}/report.txt')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task_name', type=str, default='mfq-30', help='Task to use for generating responses')
    parser.add_argument('-m', '--model_name', type=str, default='openai_chatgpt', help='Model to use for generating responses')
    parser.add_argument('-j', '--judge_model_names', nargs='+', default=['openai_chatgpt'], help='Models to use for judging responses')
    args = parser.parse_args()

    main_runner(
        args.task_name, 
        args.model_name,
        args.judge_model_names
    )