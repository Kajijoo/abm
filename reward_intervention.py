import os
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vector_model import run_vectorized_learning, run_vectorized_simulation

# ===================== CONFIG =====================
EPSILON = 0.05
STEPS = 100
N = 100
WIDTH, HEIGHT = 100, 100
N_REPS = 20

# Pre-intervention baseline (single learned state)
THETA_PRE = 0.759
P_HIGH_PRE = 1.0
P_LOW = 0.5

# Post-intervention sweep
THETA_POST_LIST = np.linspace(0.0, 1.0, 21)   # P(H)
P_HIGH_POST_LIST = np.linspace(1.0, 0.5, 21)  # lambda_H reduced to lambda_L


def worker_sim(args):
    theta_post, p_high_post, vH0, vL0, rep = args
    res = run_vectorized_simulation(
        theta=theta_post,
        epsilon=EPSILON,
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


def run_reward_intervention_surface():
    # Learn once under the high-reward baseline, then intervene.
    vH0, vL0 = run_vectorized_learning(
        p_high=P_HIGH_PRE,
        p_low=P_LOW,
        theta=THETA_PRE,
        epsilon=EPSILON,
        steps=STEPS,
        N=N,
        width=WIDTH,
        height=HEIGHT,
        seed=123,
    )

    tasks = []
    for theta_post in THETA_POST_LIST:
        for p_high_post in P_HIGH_POST_LIST:
            for rep in range(N_REPS):
                tasks.append((theta_post, p_high_post, vH0, vL0, rep))

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


def plot_delta_v_surface(df_mean, outdir):
    thetas = np.sort(df_mean["theta_post"].unique())
    p_highs = np.sort(df_mean["p_high_post"].unique())
    z = df_mean.pivot(index="p_high_post", columns="theta_post", values="delta_v").values
    t_grid, p_grid = np.meshgrid(thetas, p_highs)

    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    levels = np.linspace(z.min(), z.max(), 25)
    cs = ax.contourf(t_grid, p_grid, z, levels=levels, cmap="viridis")
    ax.contour(t_grid, p_grid, z, levels=[0.0], colors="red", linewidths=1.2)

    ax.set_xlabel(r"$P(H)$", fontsize=12)
    ax.set_ylabel(r"High-reward value ($\lambda_H$)", fontsize=12)
    ax.tick_params(labelsize=11)

    cbar = fig.colorbar(cs, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(r"$V_H - V_L$", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(os.path.join(outdir, "reward_intervention_surface.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(outdir, "reward_intervention_surface.pdf"), dpi=600, bbox_inches="tight")


def plot_lh_share_surface(df_mean, outdir):
    thetas = np.sort(df_mean["theta_post"].unique())
    p_highs = np.sort(df_mean["p_high_post"].unique())
    z_ratio = df_mean.pivot(index="p_high_post", columns="theta_post", values="LH_Ratio").values
    z_share = np.clip(z_ratio / (1.0 + z_ratio), 0.0, 1.0)
    t_grid, p_grid = np.meshgrid(thetas, p_highs)

    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    levels = np.linspace(0.0, 1.0, 21)
    cs = ax.contourf(t_grid, p_grid, z_share, levels=levels, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.contour(t_grid, p_grid, z_share, levels=[0.5], colors="red", linewidths=1.2)

    ax.set_xlabel(r"$P(H)$", fontsize=12)
    ax.set_ylabel(r"High-reward value ($\lambda_H$)", fontsize=12)
    ax.tick_params(labelsize=11)

    cbar = fig.colorbar(cs, ax=ax, shrink=0.9, pad=0.02, ticks=np.linspace(0.0, 1.0, 6))
    cbar.set_label(r"$L/(L+H)$ consumption", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(os.path.join(outdir, "reward_intervention_lh_share.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(outdir, "reward_intervention_lh_share.pdf"), dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    t0 = time.time()
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = os.path.join("resub", "reward_intervention", run_timestamp)
    os.makedirs(outdir, exist_ok=True)

    print("Running reward intervention surface...")
    vH0, vL0, df_all, df_mean = run_reward_intervention_surface()

    with open(os.path.join(outdir, "prelearning_init.txt"), "w") as f:
        f.write(
            f"Pre-learning baseline: theta={THETA_PRE}, p_high={P_HIGH_PRE}, p_low={P_LOW}, epsilon={EPSILON}\n"
        )
        f.write(f"V_H0={vH0:.6f}, V_L0={vL0:.6f}\n")

    df_all.to_csv(os.path.join(outdir, "all_agent_results.csv"), index=False)
    df_mean.to_csv(os.path.join(outdir, "reward_intervention_surface_data.csv"), index=False)

    plot_delta_v_surface(df_mean, outdir)
    plot_lh_share_surface(df_mean, outdir)

    print(f"Saved results to {outdir}")
    print(f"Total runtime: {time.time() - t0:.2f} s")
