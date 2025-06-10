import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import math
import numpy as np

# Consistent colors for all MSE lines
COLOR_MAP = {
    'Clean MSE': 'black',
    'Poisoned MSE': 'red',
    'TRIM Defense MSE': 'blue',
    'RANSAC Defense MSE': 'green',
    'Huber Defense MSE': 'orange'
}

def plot_and_save_all_models_per_dataset(csv_file, save_dir='visualisations'):
    df_results = pd.read_csv(csv_file)

    poisoning_rates = sorted(df_results['epsilon'].dropna().unique())
    models = {model: None for model in df_results['model'].dropna().unique()}
    datasets = df_results['dataset'].dropna().unique()
    attacks = df_results['attack'].dropna().unique()

    os.makedirs(save_dir, exist_ok=True)

    for dataset in datasets:
        for attack in attacks:
            df_sub = df_results[(df_results['dataset'] == dataset) & (df_results['attack'] == attack)]
            if df_sub.empty:
                continue

            n_models = len(models)
            n_cols = 2
            n_rows = math.ceil(n_models / n_cols)
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
            fig.suptitle(f'MSE vs Poisoning Rate\nDataset: {os.path.basename(dataset)} | Attack: {attack}', fontsize=16)
            axs = axs.flatten()

            for ax, model_name in zip(axs, models.keys()):
                df_model = df_sub[df_sub['model'] == model_name]
                if df_model.empty:
                    ax.set_visible(False)
                    continue

                mse_lines = {
                    'Clean MSE': [],
                    'Poisoned MSE': [],
                    'TRIM Defense MSE': [],
                    'RANSAC Defense MSE': [],
                    'Huber Defense MSE': [],
                }

                for eps in poisoning_rates:
                    row = df_model[df_model['epsilon'] == eps]
                    if not row.empty:
                        mse_lines['Clean MSE'].append(row['mse_clean'].values[0])
                        mse_lines['Poisoned MSE'].append(row['mse_poisoned'].values[0])
                        mse_lines['TRIM Defense MSE'].append(row['mse_trim'].values[0])
                        mse_lines['RANSAC Defense MSE'].append(row['mse_ransac'].values[0])
                        mse_lines['Huber Defense MSE'].append(row['mse_huber'].values[0])
                    else:
                        for k in mse_lines:
                            mse_lines[k].append(None)

                for label, values in mse_lines.items():
                    ax.plot(poisoning_rates, values, label=label, marker='o', color=COLOR_MAP[label])

                ax.set_title(model_name)
                ax.set_xlabel('Poisoning Rate (ε)')
                ax.set_ylabel('MSE')
                ax.legend()
                ax.grid(True)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            filename = f"{os.path.basename(dataset).replace('.csv', '')}_{attack}_mse.png"
            plt.savefig(os.path.join(save_dir, filename))
            plt.close(fig)
            print(f"Saved MSE plot: {filename}")

def plot_attack_times_all_models(df_results, models=['linearregression', 'ridge', 'elasticnet', 'lasso'], save_dir='visualisations'):
    attacks = df_results['attack'].dropna().unique()
    datasets = df_results['dataset'].dropna().unique()
    epsilons = sorted(df_results['epsilon'].dropna().unique())
    os.makedirs(save_dir, exist_ok=True)

    for attack in attacks:
        fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
        axs = axs.flatten()
        fig.suptitle(f'Attack Time vs Epsilon for Attack: {attack}', fontsize=16)

        for i, model in enumerate(models):
            ax = axs[i]
            df_model_attack = df_results[(df_results['attack'] == attack) & (df_results['model'].str.lower() == model)]
            if df_model_attack.empty:
                ax.set_visible(False)
                continue

            width = 0.15
            x = np.arange(len(epsilons))

            for j, dataset in enumerate(datasets):
                times = []
                for eps in epsilons:
                    row = df_model_attack[(df_model_attack['epsilon'] == eps) & (df_model_attack['dataset'] == dataset)]
                    times.append(row['attack_time_sec'].values[0] if not row.empty else 0)
                bar = ax.bar(x + j * width, times, width=width, label=os.path.basename(dataset))
                for rect in bar:
                    height = rect.get_height()
                    if height > 0:
                        ax.text(rect.get_x() + rect.get_width()/2, height, f'{height:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)

            ax.set_title(f'Model: {model.capitalize()}')
            ax.set_xlabel('Epsilon (Poisoning Rate)')
            if i % 2 == 0:
                ax.set_ylabel('Attack Time (s)')
            ax.set_xticks(x + width * (len(datasets)-1) / 2)
            ax.set_xticklabels([f'{eps:.2f}' for eps in epsilons])
            ax.legend(fontsize=8)
            ax.grid(True)

        # Hide any unused subplots if models < 4
        for ax in axs[len(models):]:
            ax.set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = f'{attack}_attack_time_all_models.png'
        plt.savefig(os.path.join(save_dir, filename))
        plt.close(fig)
        print(f"Saved combined attack time plot: {filename}")


def plot_fit_or_pred_times_all_models(df_results, time_type='fit', models=['linearregression', 'ridge', 'elasticnet', 'lasso'], save_dir='visualisations'):
    assert time_type in ['fit', 'pred']
    datasets = df_results['dataset'].dropna().unique()
    epsilons = sorted(df_results['epsilon'].dropna().unique())
    os.makedirs(save_dir, exist_ok=True)

    # Mapping to the right columns in the CSV for each defense + poisoned model
    time_cols_map = {
        'fit': {
            'poisoned': 'fit_time_poisoned',
            'trim': 'fit_time_trim',
            'ransac': 'fit_time_ransac',
            'huber': 'fit_time_huber',
        },
        'pred': {
            'poisoned': 'pred_time_poisoned',
            'trim': 'pred_time_trim',
            'ransac': 'pred_time_ransac',
            'huber': 'pred_time_huber',
        }
    }

    defense_models = ['poisoned', 'trim', 'ransac', 'huber']

    for defense in defense_models:
        fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
        axs = axs.flatten()
        fig.suptitle(f'{time_type.capitalize()} Time vs Epsilon for Defense: {defense.capitalize()}', fontsize=16)

        col_name = time_cols_map[time_type][defense]

        for i, model in enumerate(models):
            ax = axs[i]
            df_model = df_results[df_results['model'].str.lower() == model]
            if df_model.empty or col_name not in df_model.columns:
                ax.set_visible(False)
                continue

            width = 0.15
            x = np.arange(len(epsilons))

            for j, dataset in enumerate(datasets):
                times = []
                for eps in epsilons:
                    row = df_model[(df_model['epsilon'] == eps) & (df_model['dataset'] == dataset)]
                    times.append(row[col_name].values[0] if not row.empty and not pd.isna(row[col_name].values[0]) else 0)
                bar = ax.bar(x + j * width, times, width=width, label=os.path.basename(dataset))
                for rect in bar:
                    height = rect.get_height()
                    if height > 0:
                        ax.text(rect.get_x() + rect.get_width()/2, height, f'{height:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)

            ax.set_title(f'Model: {model.capitalize()}')
            ax.set_xlabel('Epsilon (Poisoning Rate)')
            if i % 2 == 0:
                ax.set_ylabel(f'{time_type.capitalize()} Time (s)')
            ax.set_xticks(x + width * (len(datasets)-1) / 2)
            ax.set_xticklabels([f'{eps:.2f}' for eps in epsilons])
            ax.legend(fontsize=8)
            ax.grid(True)

        # Hide any unused subplots if models < 4
        for ax in axs[len(models):]:
            ax.set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = f'{defense}_{time_type}_time_all_models.png'
        plt.savefig(os.path.join(save_dir, filename))
        plt.close(fig)
        print(f"Saved combined {time_type} time plot for defense '{defense}': {filename}")



def main():
    parser = argparse.ArgumentParser(description='Visualize poisoning experiment results.')
    parser.add_argument('csv_file', help='Path to the CSV file with results.')
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)

    plot_and_save_all_models_per_dataset(args.csv_file, save_dir="visualisations_mse")
    models_to_plot = ['linearregression', 'ridge', 'elasticnet', 'lasso']

    # Attack time plots combined for all models (per attack)
    plot_attack_times_all_models(df, models=models_to_plot, save_dir='visualisations_time')

    # Fit time plots combined for all defense types (poisoned + defenses)
    plot_fit_or_pred_times_all_models(df, 'fit', models=models_to_plot, save_dir='visualisations_time')

    # Prediction time plots combined for all defense types (poisoned + defenses)
    plot_fit_or_pred_times_all_models(df, 'pred', models=models_to_plot, save_dir='visualisations_time')


if __name__ == "__main__":
    main()
