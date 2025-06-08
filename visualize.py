import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_and_save_all_models_per_dataset(csv_file, save_dir='visualisations'):
    # Load CSV
    df_results = pd.read_csv(csv_file)

    # Automatically detect poisoning rates and models
    poisoning_rates = sorted(df_results['epsilon'].unique())
    models = {model: None for model in df_results['model'].unique()}

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Unique datasets and attacks
    datasets = df_results['dataset'].unique()
    attacks = df_results['attack'].unique()

    for dataset in datasets:
        for attack in attacks:
            df_sub = df_results[(df_results['dataset'] == dataset) & (df_results['attack'] == attack)]
            if df_sub.empty:
                continue

            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'MSE vs Poisoning Rate\nDataset: {os.path.basename(dataset)} | Attack: {attack}', fontsize=16)
            axs = axs.flatten()

            for ax, model_name in zip(axs, models.keys()):
                df_model = df_sub[df_sub['model'] == model_name]
                if df_model.empty:
                    ax.set_visible(False)
                    continue

                mse_clean_list = []
                mse_poisoned_list = []
                mse_trim_list = []
                mse_ransac_list = []
                mse_huber_list = []

                for eps in poisoning_rates:
                    row = df_model[df_model['epsilon'] == eps]
                    if not row.empty:
                        mse_clean_list.append(row['mse_clean'].values[0])
                        mse_poisoned_list.append(row['mse_poisoned'].values[0])
                        mse_trim_list.append(row['mse_trim'].values[0])
                        mse_ransac_list.append(row['mse_ransac'].values[0])
                        mse_huber_list.append(row['mse_huber'].values[0])
                    else:
                        mse_clean_list.append(None)
                        mse_poisoned_list.append(None)
                        mse_trim_list.append(None)
                        mse_ransac_list.append(None)
                        mse_huber_list.append(None)

                ax.plot(poisoning_rates, mse_clean_list, label='Clean MSE', marker='o')
                ax.plot(poisoning_rates, mse_poisoned_list, label='Poisoned MSE', marker='o')
                ax.plot(poisoning_rates, mse_trim_list, label='TRIM Defense MSE', marker='o')
                ax.plot(poisoning_rates, mse_ransac_list, label='RANSAC Defense MSE', marker='o')
                ax.plot(poisoning_rates, mse_huber_list, label='Huber Defense MSE', marker='o')

                ax.set_title(model_name)
                ax.set_xlabel('Poisoning Rate (ε)')
                ax.set_ylabel('MSE')
                ax.legend()
                ax.grid(True)

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # Save figure to file
            filename = f"{os.path.basename(dataset).replace('.csv', '')}_{attack}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath)
            plt.close(fig)

            print(f"Saved plot: {filepath}")

# Example usage:
plot_and_save_all_models_per_dataset('poisoning_experiment_results.csv')
