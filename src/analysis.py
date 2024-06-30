# src/analysis_utils.py
import csv
import os
from collections import Counter
import numpy as np
import tiktoken  # Make sure to install tiktoken package

def count_unique_tokens(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")  # Use appropriate tokenizer
    tokens = tokenizer.encode(text)
    return len(set(tokens))

def save_results_to_csv(results, experiment_folder):
    os.makedirs(experiment_folder, exist_ok=True)
    filename = os.path.join(experiment_folder, "results.csv")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["prompt", "temperature", "top_p", "presence_penalty", "frequency_penalty", "output"])
        for result in results:
            for output in result["outputs"]:
                writer.writerow([result["prompt"], result["temperature"], result["top_p"], result["presence_penalty"], result["frequency_penalty"], output])

def analyze_results(results, experiment_type):
    analysis = []
    for result in results:
        if experiment_type == 1:
            unique_outputs = len(set(result["outputs"]))
            analysis.append({
                "prompt": result["prompt"],
                "temperature": result["temperature"],
                "top_p": result["top_p"],
                "presence_penalty": result["presence_penalty"],
                "frequency_penalty": result["frequency_penalty"],
                "unique_outputs": unique_outputs,
                "most_common_output": Counter(result["outputs"]).most_common(1)[0][0] if result["outputs"] else "N/A",
                "most_common_count": Counter(result["outputs"]).most_common(1)[0][1] if result["outputs"] else 0
            })
        else:
            unique_token_counts = [count_unique_tokens(output) for output in result["outputs"]]
            avg_unique_tokens = np.mean(unique_token_counts)
            analysis.append({
                "prompt": result["prompt"],
                "temperature": result["temperature"],
                "top_p": result["top_p"],
                "presence_penalty": result["presence_penalty"],
                "frequency_penalty": result["frequency_penalty"],
                "avg_unique_tokens": avg_unique_tokens,
                "most_common_output": Counter(result["outputs"]).most_common(1)[0][0] if result["outputs"] else "N/A",
                "most_common_count": Counter(result["outputs"]).most_common(1)[0][1] if result["outputs"] else 0
            })
    return analysis
