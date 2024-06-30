# src/visualization_utils.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_results(analysis, n, title, experiment_type, experiment_folder):
    os.makedirs(experiment_folder, exist_ok=True)
    prompts_set = set([entry["prompt"] for entry in analysis])

    for prompt in prompts_set:
        data = [entry for entry in analysis if entry["prompt"] == prompt]
        
        if experiment_type == 1:
            # For Experiment 1, temperature on the y axis and top_p on the x axis
            temperatures = sorted(set(entry["temperature"] for entry in data))
            top_ps = sorted(set(entry["top_p"] for entry in data))
            unique_outputs_matrix = np.zeros((len(temperatures), len(top_ps)))

            for entry in data:
                t_idx = temperatures.index(entry["temperature"])
                tp_idx = top_ps.index(entry["top_p"])
                unique_outputs_matrix[t_idx, tp_idx] = entry["unique_outputs"]

            y_labels = [f'T: {t}' for t in temperatures]
            x_labels = [f'Tp: {tp}' for tp in top_ps]

            vmin = np.min(unique_outputs_matrix)
            vmax = np.max(unique_outputs_matrix)

            plt.figure(figsize=(15, 10))
            ax = sns.heatmap(unique_outputs_matrix, annot=True, fmt='.2f', xticklabels=x_labels, yticklabels=y_labels, cmap="YlGnBu", vmin=vmin, vmax=vmax)

            plt.xlabel('Top-p')
            plt.ylabel('Temperature')

        elif experiment_type == 2:
            # For Experiment 2, temperature and top_p combined in the y axis (only where temp == top_p) and presence on the x axis
            temp_top_p_combinations = sorted(set((entry["temperature"], entry["top_p"]) for entry in data if entry["temperature"] == entry["top_p"]))
            presence_penalties = sorted(set(entry["presence_penalty"] for entry in data))
            unique_tokens_matrix = np.zeros((len(temp_top_p_combinations), len(presence_penalties)))

            for entry in data:
                if entry["temperature"] == entry["top_p"]:
                    tt_idx = temp_top_p_combinations.index((entry["temperature"], entry["top_p"]))
                    pp_idx = presence_penalties.index(entry["presence_penalty"])
                    unique_tokens_matrix[tt_idx, pp_idx] = entry["avg_unique_tokens"]

            y_labels = [f'T: {tt[0]}, Tp: {tt[1]}' for tt in temp_top_p_combinations]
            x_labels = [str(pp) for pp in presence_penalties]

            vmin = np.min(unique_tokens_matrix)
            vmax = np.max(unique_tokens_matrix)

            plt.figure(figsize=(15, 10))
            ax = sns.heatmap(unique_tokens_matrix, annot=True, fmt='.2f', xticklabels=x_labels, yticklabels=y_labels, cmap="YlGnBu", vmin=vmin, vmax=vmax)

            plt.xlabel('Presence Penalty')
            plt.ylabel('Temperature and Top-p Combination')

        elif experiment_type == 3:
            # For Experiment 3, temperature and top_p combined in the y axis (only where temp == top_p) and frequency on the x axis
            temp_top_p_combinations = sorted(set((entry["temperature"], entry["top_p"]) for entry in data if entry["temperature"] == entry["top_p"]))
            frequency_penalties = sorted(set(entry["frequency_penalty"] for entry in data))
            unique_tokens_matrix = np.zeros((len(temp_top_p_combinations), len(frequency_penalties)))

            for entry in data:
                if entry["temperature"] == entry["top_p"]:
                    tt_idx = temp_top_p_combinations.index((entry["temperature"], entry["top_p"]))
                    fp_idx = frequency_penalties.index(entry["frequency_penalty"])
                    unique_tokens_matrix[tt_idx, fp_idx] = entry["avg_unique_tokens"]

            y_labels = [f'T: {tt[0]}, Tp: {tt[1]}' for tt in temp_top_p_combinations]
            x_labels = [str(fp) for fp in frequency_penalties]

            vmin = np.min(unique_tokens_matrix)
            vmax = np.max(unique_tokens_matrix)

            plt.figure(figsize=(15, 10))
            ax = sns.heatmap(unique_tokens_matrix, annot=True, fmt='.2f', xticklabels=x_labels, yticklabels=y_labels, cmap="YlGnBu", vmin=vmin, vmax=vmax)

            plt.xlabel('Frequency Penalty')
            plt.ylabel('Temperature and Top-p Combination')

        elif experiment_type == 4:
            # For Experiment 4, temperature and top_p combined in the y axis (only where temp == top_p) and presence and frequency combined on the x axis (only where presence == frequency)
            temp_top_p_combinations = sorted(set((entry["temperature"], entry["top_p"]) for entry in data if entry["temperature"] == entry["top_p"]))
            penalty_combinations = sorted(set((entry["frequency_penalty"], entry["presence_penalty"]) for entry in data if entry["frequency_penalty"] == entry["presence_penalty"]))
            unique_tokens_matrix = np.zeros((len(temp_top_p_combinations), len(penalty_combinations)))

            for entry in data:
                if entry["temperature"] == entry["top_p"] and entry["frequency_penalty"] == entry["presence_penalty"]:
                    tt_idx = temp_top_p_combinations.index((entry["temperature"], entry["top_p"]))
                    pc_idx = penalty_combinations.index((entry["frequency_penalty"], entry["presence_penalty"]))
                    unique_tokens_matrix[tt_idx, pc_idx] = entry["avg_unique_tokens"]

            y_labels = [f'T: {tt[0]}, Tp: {tt[1]}' for tt in temp_top_p_combinations]
            x_labels = [f'FP: {pc[0]}, PP: {pc[1]}' for pc in penalty_combinations]

            vmin = np.min(unique_tokens_matrix)
            vmax = np.max(unique_tokens_matrix)

            plt.figure(figsize=(15, 10))
            ax = sns.heatmap(unique_tokens_matrix, annot=True, fmt='.2f', xticklabels=x_labels, yticklabels=y_labels, cmap="YlGnBu", vmin=vmin, vmax=vmax)

            plt.xlabel('Frequency Penalty and Presence Penalty Combinations')
            plt.ylabel('Temperature and Top-p Combination')

        plt.figtext(0.5, 0.92, f'Hyperparameter Experiment with GPT-4o', ha='center', fontsize=12, fontweight='bold')
        plt.figtext(0.5, 0.86, title, ha='center', fontsize=10)
        plt.figtext(0.5, 0.90, f'Prompt: "{prompt}"', ha='center', fontsize=10, wrap=True)
        plt.figtext(0.5, 0.88, f'Number of runs: {n}', ha='center', fontsize=10)
        plt.figtext(0.5, 0.84, f'{"Average Unique Tokens" if experiment_type != 1 else "Unique Outputs"}', ha='center', fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.85])  # Adjust to fit the figtexts at the top
        heatmap_path = os.path.join(experiment_folder, f'heatmap_experiment_{experiment_type}.png')
        plt.savefig(heatmap_path)
        plt.show()
