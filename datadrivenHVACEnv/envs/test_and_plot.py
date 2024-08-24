import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions(env, episode, actuals_df, predictions_df, save_to_file=False):
    
    if save_to_file:
        actuals_df.to_csv(f"env_logs/actual_df_{episode}.csv")
        predictions_df.to_csv(f"env_logs/predictions_df{episode}.csv")
    
    for col in self.state_variables+["Gas_volume"]:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(actuals_df.index, actuals_df[col], label="Actual", color="tab:blue")
        ax.plot(
            actuals_df.index,
            predictions_df[col],
            label="Predicted",
            linestyle="--",
            color="tab:orange",
        )
        ax.set_title(f'Actual vs Predicted Values for {col}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Values')
        ax.legend()

        plt.savefig(f'plots/{col}_plot.png')
        plt.close(fig) 
        
        
def plot_results(env, episode, rewards):
    pass