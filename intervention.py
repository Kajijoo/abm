import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
from multiprocessing import Pool
import time
import os, warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from vector_model import run_vectorized_learning, run_vectorized_simulation

mpl.rcParams['font.family'] = 'Arial'

# Timestamped root folder
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ---------------------------------------------------------------------
# Learning phase
# ---------------------------------------------------------------------
def run_learning(p_high, p_low, steps=100, theta=0.759, epsilon=0.0, N=10000, seed=123):
    np.random.seed(seed)
    vH, vL = run_vectorized_learning(
        p_high=p_high, p_low=p_low, theta=theta, epsilon=epsilon,
        steps=steps, N=N, width=100, height=N, seed=seed
    )
    return vH, vL


# ---------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------
def run_single_sim(args):
    theta, epsilon, seed, p_high, p_low, vhigh0, vlow0 = args
    np.random.seed(seed)

    res = run_vectorized_simulation(
        theta=theta,
        epsilon=epsilon,
        p_high=p_high,
        p_low=p_low,
        vhigh0=vhigh0,
        vlow0=vlow0,
        steps=100,
        N=100,
        width=100,
        height=100,
        seed=seed
    )

    return (
        theta,
        epsilon,
        res["delta_V"],
        res["Value_High"],
        res["Value_Low"],
        res["LH_Ratio"]
    )


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def plot_lh_ratio_heatmap(df, outdir):
    thetas = np.sort(df["theta"].unique())
    eps = np.sort(df["epsilon"].unique())
    Z_ratio = df.pivot(index="epsilon", columns="theta", values="LH_Ratio").values
    Z = Z_ratio / (1.0 + Z_ratio)
    Z = np.clip(Z, 0.0, 1.0)
    T, E = np.meshgrid(thetas, eps)

    fig, ax = plt.subplots(figsize=(4.0, 3.2), constrained_layout=True)
    levels = np.linspace(0.0, 1.0, 21)
    cs = ax.contourf(T, E, Z, levels=levels, cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xlabel(r'$P(H)$', fontsize=12)
    ax.set_ylabel(r'$\epsilon$', fontsize=12)
    ax.tick_params(labelsize=11)

    cbar = fig.colorbar(cs, ax=ax, shrink=0.9, pad=0.02, ticks=np.linspace(0.0, 1.0, 6))
    cbar.set_label(rf'$L/(L+H)$ consumption', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(f"{outdir}/lh_ratio_heatmap.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{outdir}/lh_ratio_heatmap.pdf", dpi=600, bbox_inches="tight")



def plot_combined(df1, df2, p1, p2):
    os.makedirs(f"resub/interventions/{RUN_TIMESTAMP}", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True, constrained_layout=True)

    plots = [
        (axes[0], df1, p1, '150%'),
        (axes[1], df2, p2, '200%')
    ]

    cmap = plt.cm.viridis
    vmin_all, vmax_all = -0.5, 1.0
    norm_all = mpl.colors.Normalize(vmin=vmin_all, vmax=vmax_all)
    levels = np.linspace(vmin_all, vmax_all, 21)

    for ax, df, (p_high, p_low), title in plots:
        thetas = np.sort(df["theta"].unique())
        eps = np.sort(df["epsilon"].unique())
        Z = df.pivot(index="epsilon", columns="theta", values="delta_v").values
        T, G = np.meshgrid(thetas, eps)

        cs = ax.contourf(T, G, Z, levels=levels, cmap=cmap, norm=norm_all)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r'$P(H)$', fontsize=12)
        ax.tick_params(labelsize=11)
        xticks = np.linspace(0.0, 1.0, 6)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1f}" for x in xticks])

        if ax is axes[0]:
            ax.set_ylabel(r'$\epsilon$', fontsize=12)

        # Add ΔV=0 contour line (red)
        ax.contour(T, G, Z, levels=[0], colors='red', linewidths=1.2)


    # Shared colorbar
    cbar = fig.colorbar(cs, ax=axes, orientation='vertical', shrink=0.85,
                        pad=0.04, ticks=[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label(r'$V_H - V_L$', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    plt.savefig(f"resub/interventions/{RUN_TIMESTAMP}/interventions.png",
                dpi=600, bbox_inches='tight')
    plt.savefig(f"resub/interventions/{RUN_TIMESTAMP}/interventions.pdf",
                dpi=600, bbox_inches='tight')


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------
def run_experiment(p_high=0.75, p_low=0.5, tag="baseline"):
    vhigh0, vlow0 = run_learning(p_high, p_low, steps=100, theta=3.0, epsilon=0)
    print(f"Pre-learning ({p_high}, {p_low}) -> V_H0={vhigh0:.4f}, V_L0={vlow0:.4f}")

    thetas =    np.linspace(0, 1, 21)
    epsilons =  np.linspace(0, 1, 21)
    n_reps = 20

    args = [(theta, epsilon, seed, p_high, p_low, vhigh0, vlow0)
            for theta in thetas for epsilon in epsilons for seed in range(n_reps)]

    print(f"Running {len(args)} simulations for p_high={p_high}, p_low={p_low}...")
    with Pool() as pool:
        results = pool.map(run_single_sim, args)

    df = pd.DataFrame(results, columns=["theta", "epsilon", "delta_v",
                                        "Value_High", "Value_Low", "LH_Ratio"])
    df_mean = df.groupby(["theta", "epsilon"], as_index=False).mean()

    outdir = f"resub/interventions/{RUN_TIMESTAMP}/{tag}"
    os.makedirs(outdir, exist_ok=True)

    with open(f"{outdir}/prelearning_init.txt", "w") as f:
        f.write(f"p_high={p_high}, p_low={p_low}")
        f.write(f"V_H0={vhigh0:.6f}, V_L0={vlow0:.6f}\n")

    df.to_csv(f"{outdir}/all_agent_results.csv", index=False)
    df_mean.to_csv(f"{outdir}/deltaV_contour_data.csv", index=False)

    plot_lh_ratio_heatmap(df_mean, outdir)
    print(f"Saved results to {outdir}")
    return df_mean

# ---------------------------------------------------------------------
# Entry point with total runtime
# ---------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()

    df1 = run_experiment(p_high=0.75, p_low=0.5, tag="baseline")
    df2 = run_experiment(p_high=1.0, p_low=0.5, tag="strong_diff")
    plot_combined(df1, df2, (0.75, 0.5), (1.0, 0.5))

    total_time = time.time() - start_time
    print(f"\n=== Total runtime: {total_time/60:.2f} minutes ===")
