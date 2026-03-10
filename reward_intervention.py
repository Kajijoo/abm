import os
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

from vector_model import run_vectorized_learning, run_vectorized_simulation

# ===================== CONFIG =====================
EPSILON_POST = 0.05
EPSILON_PRE = 0.05
STEPS = 100
N = 100
WIDTH, HEIGHT = 100, 100
N_REPS = 20
THETA_PRE = 0.759
P_LOW = 0.5

# Two pre-learning reward baselines (relative to p_low=0.5)
REWARD_SCENARIOS = [
    ("diff_150", 0.75, "150%"),
    ("diff_200", 1.00, "200%"),
]

# Post-intervention sweep
THETA_POST_LIST = np.linspace(0.0, 1.0, 21)   # P(H)
P_HIGH_POST_LIST = np.linspace(1.0, 0.5, 21)  # lambda_H reduced to lambda_L

mpl.rcParams["font.family"] = "Arial"


def worker_sim(args):
    theta_post, p_high_post, vH0, vL0, rep = args
    res = run_vectorized_simulation(
        theta=theta_post,
        epsilon=EPSILON_POST,
        p_high=p_high_post,
        p_low=P_LOW,
        vhigh0=vH0,
        vlow0=vL0,
        steps=STEPS,
        N=N,
        width=WIDTH,
        height=HEIGHT,
        seed=rep,
    )
    return (
        theta_post,
        p_high_post,
        res["delta_V"],
        res["Value_High"],
        res["Value_Low"],
        res["LH_Ratio"],
    )


def run_reward_intervention_surface(pre_p_high):
    # Learn once under each baseline reward setup, then apply intervention sweep.
    vH0, vL0 = run_vectorized_learning(
        p_high=pre_p_high,
        p_low=P_LOW,
        theta=THETA_PRE,
        epsilon=EPSILON_PRE,
        steps=STEPS,
        N=N,
        width=WIDTH,
        height=HEIGHT,
        seed=123,
    )

    tasks = [
        (theta_post, p_high_post, vH0, vL0, rep)
        for theta_post in THETA_POST_LIST
        for p_high_post in P_HIGH_POST_LIST
        for rep in range(N_REPS)
    ]

    procs = min(max(1, cpu_count() - 1), 12)
    chunksize = max(1, len(tasks) // (procs * 8))

    with Pool(processes=procs) as pool:
        out = pool.map(worker_sim, tasks, chunksize=chunksize)

    df = pd.DataFrame(
        out,
        columns=[
            "theta_post",
            "p_high_post",
            "delta_v",
            "Value_High",
            "Value_Low",
            "LH_Ratio",
        ],
    )
    df_mean = df.groupby(["theta_post", "p_high_post"], as_index=False).mean()
    return vH0, vL0, df, df_mean


def _mesh(df_mean, value_col):
    thetas = np.sort(df_mean["theta_post"].unique())
    p_highs = np.sort(df_mean["p_high_post"].unique())
    z = df_mean.pivot(index="p_high_post", columns="theta_post", values=value_col).values
    t_grid, p_grid = np.meshgrid(thetas, p_highs)
    return t_grid, p_grid, z


def _zero_centered_norm(*arrays):
    """Build a norm that always maps 0.0 to the center of the colormap."""
    vmin = min(float(np.nanmin(arr)) for arr in arrays)
    vmax = max(float(np.nanmax(arr)) for arr in arrays)
    if vmin >= 0.0:
        vmin = -1e-9
    if vmax <= 0.0:
        vmax = 1e-9
    return mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)


def plot_delta_v_surface(df_mean, outdir):
    t_grid, p_grid, z = _mesh(df_mean, "delta_v")
    norm = _zero_centered_norm(z)

    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    levels = np.linspace(norm.vmin, norm.vmax, 21)
    cs = ax.contourf(t_grid, p_grid, z, levels=levels, cmap="coolwarm", norm=norm)
    ax.contour(t_grid, p_grid, z, levels=[0.0], colors="red", linewidths=1.2)

    ax.set_xlabel(r"$P(H)$", fontsize=12)
    ax.set_ylabel(r"$\lambda_H$", fontsize=12)
    ax.tick_params(labelsize=11)

    cbar = fig.colorbar(
        cs,
        ax=ax,
        shrink=0.9,
        pad=0.02,
        ticks=[-0.5, 0.0, 0.5, 0,9],
    )
    cbar.set_label(r"$V_H - V_L$", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(os.path.join(outdir, "reward_intervention_surface.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(outdir, "reward_intervention_surface.pdf"), dpi=600, bbox_inches="tight")


def plot_lh_share_surface(df_mean, outdir):
    t_grid, p_grid, z_ratio = _mesh(df_mean, "LH_Ratio")
    z_share = np.clip(z_ratio / (1.0 + z_ratio), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    levels = np.linspace(0.0, 1.0, 21)
    cs = ax.contourf(t_grid, p_grid, z_share, levels=levels, cmap="coolwarm_r", vmin=0.0, vmax=1.0)
    ax.contour(t_grid, p_grid, z_share, levels=[0.5], colors="red", linewidths=1.2)

    ax.set_xlabel(r"$P(H)$", fontsize=12)
    ax.set_ylabel(r"$\lambda_H$", fontsize=12)
    ax.tick_params(labelsize=11)

    cbar = fig.colorbar(cs, ax=ax, shrink=0.9, pad=0.02, ticks=np.linspace(0.0, 1.0, 6))
    cbar.set_label(r"$L/(L+H)$ consumption", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(os.path.join(outdir, "reward_intervention_lh_share.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(outdir, "reward_intervention_lh_share.pdf"), dpi=600, bbox_inches="tight")


def plot_delta_v_combined(df_150, df_200, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1), sharey=True, constrained_layout=True)
    datasets = [(axes[0], df_150, "150%"), (axes[1], df_200, "200%")]
    z_150 = _mesh(df_150, "delta_v")[2]
    z_200 = _mesh(df_200, "delta_v")[2]
    norm = _zero_centered_norm(z_150, z_200)
    levels = np.linspace(norm.vmin, norm.vmax, 21)

    for ax, df_mean, title in datasets:
        t_grid, p_grid, z = _mesh(df_mean, "delta_v")
        cs = ax.contourf(t_grid, p_grid, z, levels=levels, cmap="coolwarm", norm=norm)
        ax.contour(t_grid, p_grid, z, levels=[0.0], colors="red", linewidths=1.2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r"$P(H)$", fontsize=12)
        ax.tick_params(labelsize=11)
        if ax is axes[0]:
            ax.set_ylabel(r"$\lambda_H$", fontsize=12)

    cbar = fig.colorbar(
        cs,
        ax=axes,
        orientation="vertical",
        shrink=0.85,
        pad=0.04,
        ticks=[-0.5, 0.0, 0.5, 1.0],
    )
    cbar.set_label(r"$V_H - V_L$", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    plt.savefig(os.path.join(outdir, "reward_intervention_surface_combined.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(outdir, "reward_intervention_surface_combined.pdf"), dpi=600, bbox_inches="tight")


def plot_lh_share_combined(df_150, df_200, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1), sharey=True, constrained_layout=True)
    datasets = [(axes[0], df_150, "150%"), (axes[1], df_200, "200%")]
    levels = np.linspace(0.0, 1.0, 21)

    for ax, df_mean, title in datasets:
        t_grid, p_grid, z_ratio = _mesh(df_mean, "LH_Ratio")
        z_share = np.clip(z_ratio / (1.0 + z_ratio), 0.0, 1.0)
        cs = ax.contourf(t_grid, p_grid, z_share, levels=levels, cmap="coolwarm_r", vmin=0.0, vmax=1.0)
        ax.contour(t_grid, p_grid, z_share, levels=[0.5], colors="red", linewidths=1.2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r"$P(H)$", fontsize=12)
        ax.tick_params(labelsize=11)
        if ax is axes[0]:
            ax.set_ylabel(r"$\lambda_H$", fontsize=12)

    cbar = fig.colorbar(
        cs,
        ax=axes,
        orientation="vertical",
        shrink=0.85,
        pad=0.04,
        ticks=np.linspace(0.0, 1.0, 6),
    )
    cbar.set_label(r"$L/(L+H)$ consumption", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    plt.savefig(os.path.join(outdir, "reward_intervention_lh_share_combined.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(outdir, "reward_intervention_lh_share_combined.pdf"), dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    t0 = time.time()
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root_outdir = os.path.join("resub", "reward_intervention", run_timestamp)
    os.makedirs(root_outdir, exist_ok=True)

    scenario_means = {}

    for tag, p_high_pre, label in REWARD_SCENARIOS:
        print(f"Running reward intervention surface for {label} baseline...")
        outdir = os.path.join(root_outdir, tag)
        os.makedirs(outdir, exist_ok=True)

        vH0, vL0, df_all, df_mean = run_reward_intervention_surface(p_high_pre)
        scenario_means[label] = df_mean

        with open(os.path.join(outdir, "prelearning_init.txt"), "w") as f:
            f.write(
                f"Pre-learning baseline: theta={THETA_PRE}, p_high={p_high_pre}, p_low={P_LOW}, epsilon={EPSILON_PRE}\\n"
            )
            f.write(f"Post-learning epsilon={EPSILON_POST}\\n")
            f.write(f"V_H0={vH0:.6f}, V_L0={vL0:.6f}\\n")

        df_all.to_csv(os.path.join(outdir, "all_agent_results.csv"), index=False)
        df_mean.to_csv(os.path.join(outdir, "reward_intervention_surface_data.csv"), index=False)

        plot_delta_v_surface(df_mean, outdir)
        plot_lh_share_surface(df_mean, outdir)

    plot_delta_v_combined(scenario_means["150%"], scenario_means["200%"], root_outdir)
    plot_lh_share_combined(scenario_means["150%"], scenario_means["200%"], root_outdir)

    print(f"Saved results to {root_outdir}")
    print(f"Total runtime: {time.time() - t0:.2f} s")
