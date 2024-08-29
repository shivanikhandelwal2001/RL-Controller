import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions(
    env,
    episode,
    actuals,
    predictions,
    state_variables,
    action_variables,
    save_to_file=False,
):
    actuals_df = pd.DataFrame(actuals, columns=state_variables + action_variables)
    predictions_df = pd.DataFrame(
        predictions, columns=state_variables + action_variables
    )

    if save_to_file:
        actuals_df.to_csv(
            f"env_logs/episode_actual_data/actual_df_{episode}.csv", index=False
        )
        predictions_df.to_csv(
            f"env_logs/episode_prediction_data/predictions_df{episode}.csv", index=False
        )

        for col in actuals_df.columns:
            fig, ax = plt.subplots(figsize=(20, 15))
            ax.plot(actuals_df.index, actuals_df[col], label="Actual", color="tab:blue")
            ax.plot(
                actuals_df.index,
                predictions_df[col],
                label="Predicted",
                linestyle="--",
                color="tab:orange",
            )
            ax.set_title(f"Actual vs Predicted Values for {col}")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Values")
            ax.legend()

            dir_path = f"env_logs/plots/{episode}_plots"
            os.makedirs(dir_path, exist_ok=True)

            plt.savefig(f"{dir_path}/{col}.png")
            plt.close(fig)


def plot_results(env, episode, rewards):
    pass
