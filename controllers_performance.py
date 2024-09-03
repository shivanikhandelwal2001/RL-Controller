import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

bang_bang_df = pd.read_csv("final_results/bb_episode_predictions.csv")
rl_df = pd.read_csv("final_results/lr_episode_predictions.csv")

# 1. Plot temperature vs. time for each space
temp_columns = [col for col in bang_bang_df.columns if col.endswith("Air_temperature")]

for col in temp_columns:
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.plot(bang_bang_df.index, bang_bang_df[col], label="Bang-Bang", color="tab:blue")
    ax.plot(
        rl_df.index,
        rl_df[col],
        label="Predicted",
        linestyle="--",
        color="tab:orange",
    )
    ax.set_title(f"Temperature vs. Time for {col}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()

    plt.savefig(f"final_results/plots/{col}_temp.png")
    plt.close(fig)


# 2. Plot cumulative energy consumption for both controllers
bang_bang_df["CumulativeEnergy"] = bang_bang_df["Gas_volume"].cumsum()
rl_df["CumulativeEnergy"] = rl_df["Gas_volume"].cumsum()

plt.figure(figsize=(20, 15))
sns.lineplot(
    data=bang_bang_df,
    x=bang_bang_df.index,
    y="CumulativeEnergy",
    label="Bang-Bang",
    color="blue",
)
sns.lineplot(
    data=rl_df, x=rl_df.index, y="CumulativeEnergy", label="RL", color="orange"
)
plt.title("Cumulative Energy Consumption for Bang-Bang vs. RL Controllers")
plt.xlabel("Time Steps")
plt.ylabel("Cumulative Energy Consumption")
plt.legend(title="Controller")

plt.savefig(f"final_results/plots/cumulative_cost.png")
plt.close()


# 3. Bar graph showing min, max, and avg temperature for each space and controller
bang_bang_summary = (
    bang_bang_df[temp_columns].describe().loc[["min", "max", "mean"]].T.reset_index()
)
bang_bang_summary.rename(
    columns={
        "index": "Space",
        "min": "min_temp",
        "max": "max_temp",
        "mean": "avg_temp",
    },
    inplace=True,
)
bang_bang_summary["Controller"] = "Bang-Bang"

rl_summary = rl_df[temp_columns].describe().loc[["min", "max", "mean"]].T.reset_index()
rl_summary.rename(
    columns={
        "index": "Space",
        "min": "min_temp",
        "max": "max_temp",
        "mean": "avg_temp",
    },
    inplace=True,
)
rl_summary["Controller"] = "RL"

summary_df = pd.concat([bang_bang_summary, rl_summary])

spaces = summary_df["Space"].unique()
temp_types = ["min_temp", "max_temp", "avg_temp"]
controllers = summary_df["Controller"].unique()

bar_width = 0.2
space_between_stats_groups = 0.1
space_between_groups = 0.4
x = np.arange(len(spaces)) * (
    len(temp_types) * len(controllers) * (bar_width)
    + (len(temp_types) - 1) * space_between_stats_groups
    + space_between_groups
)

fig, ax = plt.subplots(figsize=(20, 15))
colors = cm.get_cmap("Set1", len(temp_types) * len(controllers))

for i, temp_type in enumerate(temp_types):
    for j, controller in enumerate(controllers):
        df_subset = summary_df[summary_df["Controller"] == controller]

        bar_positions = (
            x
            + i * (len(controllers) * bar_width + space_between_stats_groups)
            + j * bar_width
        )
        ax.bar(
            bar_positions,
            df_subset[temp_type],
            width=bar_width,
            color=colors(i * len(controllers) + j),
            label=f"{controller} - {temp_type}",
        )

ax.set_xticks(
    x
    + (len(controllers) * bar_width + space_between_stats_groups)
    * (len(temp_types) - 1)
    / 2
)
ax.set_xticklabels(spaces, rotation=45, ha="right")
ax.set_xlabel("Space")
ax.set_ylabel("Temperature")
ax.set_title("Temperature Comparison for Spaces by Controller Type")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()

plt.savefig(f"final_results/plots/statistics_comfort.png")
plt.close()
