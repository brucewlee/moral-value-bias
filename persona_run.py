from api_keys import set_api_keys
set_api_keys()
import os, logging, sys
logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.CRITICAL)
from nutcracker.data import Task, Pile, InstanceCollection
from nutcracker.runs import Schema
from nutcracker.evaluator import AutoEvaluator, mfq_30_generate_report, pvq_rr_generate_report
import argparse
from utils import generate_persona_description, model_selector
import numpy as np
import random



def main_runner(task_name, model_name, personas_per_question, seed, judge_model_names):
    task = Task.load_from_db(
        task_name = task_name, 
        db_directory ='../nutcracker-db/db'
    )
    # sample 5 for demo
    # task.sample(3, in_place=True)
    persona_task = InstanceCollection()
    
    personas = generate_persona_description(personas_per_question)
    persona_idx_list = []
    question_number_list = []
    for instance in task:
        for idx, persona in enumerate(personas):
            new_instance = instance.copy()  # Create a new copy of the instance
            prompt = "Let's role-play. I will ask you a question and you must give me an answer. I want you to act as the person described below. Think from the person's perspective. \n\n"
            prompt += f"{persona['description']} \n\n"
            prompt += "Use the given information to answer the question below. \n\n"
            prompt += new_instance.user_prompt
            new_instance.user_prompt = prompt
            new_instance.persona = persona
            new_instance.persona_idx = idx
            question_number_list.append(instance.question_number)
            persona_idx_list.append(idx)
            persona_task.instances.append(new_instance)

    # running this experiment updates each instance's model_response property in truthfulqa data object with ChatGPT responses
    experiment = Schema(
        model = model_selector(model_name),
        data = persona_task
        )
    experiment.run()

    for judge_model_name in judge_model_names:
        # running this evaluation updates each instance's response_correct property in truthfulqa data object with evaluations
        evaluation = AutoEvaluator(
            model = model_selector(judge_model_name),
            data = persona_task
            )
        evaluation.run()
        
        question_number_set = set(question_number_list)

        interval = 10
        for sample_personas_per_question in range(interval, personas_per_question + 1, interval):
            sampled_persona_task = InstanceCollection()
            sampled_persona_idx_list = random.sample(persona_idx_list, sample_personas_per_question)
            
            for question_number in question_number_set:
                for persona_idx in sampled_persona_idx_list:
                    for instance in persona_task.instances:
                        if instance.question_number == question_number and instance.persona_idx == persona_idx:
                            new_instance = instance.copy()
                            sampled_persona_task.instances.append(new_instance)
                            break

            # Create a new directory based on the command-line arguments
            output_dir = f"runs/persona/{task_name}_{model_name}/seed{seed}_ppq{sample_personas_per_question}/judged_by_{judge_model_name}"
            os.makedirs(output_dir, exist_ok=True)

            sampled_persona_task.save_records(f"{output_dir}/records.jsonl", keys = ["question_number", "centerpiece", "options", "persona", "persona_idx", "judge_interpretation", "user_prompt", "model_response"])

            if task_name == "mfq-30":
                mfq_30_generate_report(sampled_persona_task, f'{output_dir}/report.txt')
            elif task_name == "pvq-rr":
                pvq_rr_generate_report(sampled_persona_task, f'{output_dir}/report.txt')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task_name', type=str, default='mfq-30', help='Task to use for generating responses')
    parser.add_argument('-m', '--model_name', type=str, default='openai_chatgpt', help='Model to use for generating responses')
    parser.add_argument('-p', '--personas_per_question', type=int, default=21, help='Number of personas to generate per question')
    parser.add_argument('-s', '--seed', type=int, default=2, help='Seed for random persona generation')
    parser.add_argument('-j', '--judge_model_names', nargs='+', default=['openai_chatgpt'], help='Models to use for judging responses')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    main_runner(
        args.task_name, 
        args.model_name,
        args.personas_per_question,
        args.seed,
        args.judge_model_names
    )