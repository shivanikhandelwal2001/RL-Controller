import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_predictions(
    env,
    episode,
    actuals,
    predictions,
    state_variables,
    action_variables,
    save_to_file="env_logs",
):
    save_actual_dir = f"{save_to_file}/episode_actual_data"
    os.makedirs(save_actual_dir, exist_ok=True)

    save_prediction_dir = f"{save_to_file}/episode_prediction_data"
    os.makedirs(save_prediction_dir, exist_ok=True)

    dir_path = f"{save_to_file}/plots/{episode}_plots"
    os.makedirs(dir_path, exist_ok=True)

    actuals_df = pd.DataFrame(actuals, columns=state_variables + action_variables)
    predictions_df = pd.DataFrame(
        predictions, columns=state_variables + action_variables
    )

    actuals_df.to_csv(f"{save_actual_dir}/actual_df_{episode}.csv", index=False)
    predictions_df.to_csv(
        f"{save_prediction_dir}/predictions_df{episode}.csv", index=False
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

        plt.savefig(f"{dir_path}/{col}.png")
        plt.close(fig)


def plot_results(env, episode, rewards):
    pass


def plot_discomfort_function(
    setpoint=21,
    temp_range=(15, 27),
    title="Discomfort Function for Temperature Control",
):
    """
    Plots the discomfort function for a given setpoint temperature.
    """

    x = np.linspace(temp_range[0], temp_range[1], 100)

    def discomfort(x):
        return (x - setpoint) ** 2

    y = discomfort(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f"Discomfort = (x - {setpoint})^2", color="blue")
    plt.axvline(
        x=setpoint,
        color="red",
        linestyle="--",
        label=f"Setpoint Temperature ({setpoint}°C)",
    )
    plt.axhline(y=0, color="black", linestyle="-")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Discomfort Level")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig("final_results/discomfort_function.png")


plot_discomfort_function()
