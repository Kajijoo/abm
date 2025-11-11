import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Pool
import warnings
import os
from datetime import datetime
from model import LearningModel

warnings.simplefilter(action='ignore', category=FutureWarning)

RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

mpl.rcParams['font.family'] = 'Arial'

def run_learning(p_high, p_low, steps=100, theta=3.0, epsilon=0.05, N=1000, seed=123):
    np.random.seed(seed)
    model = LearningModel(
        N=N,
        width=100,
        height=1000,
        learning_model='RWE',
        theta=theta,
        epsilon=epsilon
    )

    for a in model.agents:
        a.p_high = p_high
        a.p_low = p_low

    for _ in range(steps):
        model.step()

    vals_high = float(np.mean([a.value_high for a in model.agents]))
    vals_low = float(np.mean([a.value_low for a in model.agents]))

    return vals_high, vals_low

def run_single_sim(args):
    theta, epsilon, seed, p_high, p_low, vhigh0, vlow0 = args
    np.random.seed(seed)

    steps = 100
    N = 1000

    model = LearningModel(
        N=N,
        width=100,
        height=N,
        learning_model='RWE',
        theta=theta,
        epsilon=epsilon
    )

    for a in model.agents:
        a.epsilon = epsilon
        a.p_high = p_high
        a.p_low = p_low
        a.value_high = vhigh0
        a.value_low = vlow0

    for _ in range(steps):
        model.step()

    vhighs = [a.value_high for a in model.agents]
    vlows = [a.value_low for a in model.agents]
    deltas = [vh - vl for vh, vl in zip(vhighs, vlows)]

    l_counts = [a.foods_consumed["L"] for a in model.agents]
    h_counts = [a.foods_consumed["H"] for a in model.agents]

    ratios = []
    for Lc, Hc in zip(l_counts, h_counts):
        if Hc == 0:
            ratios.append(np.nan)
        else:
            ratios.append(Lc / Hc)

    mean_ratio = float(np.nanmean(ratios))

    return (
        theta,
        epsilon,
        float(np.mean(deltas)),
        float(np.mean(vhighs)),
        float(np.mean(vlows)),
        mean_ratio
    )

def plot_lh_ratio_3d(df, outdir):
    thetas = sorted(df["theta"].unique())
    eps = sorted(df["epsilon"].unique())

    T, E = np.meshgrid(thetas, eps)

    Z_ratio = df.pivot(index="epsilon", columns="theta", values="LH_Ratio").values
    Z_delta = df.pivot(index="epsilon", columns="theta", values="delta_v").values

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection="3d")

    norm = plt.Normalize(Z_delta.min(), Z_delta.max())
    colors = plt.cm.viridis(norm(Z_delta))

    surf = ax.plot_surface(T, E, Z_ratio, facecolors=colors, edgecolor="none")

    mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array(Z_delta)
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label="delta V")

    ax.set_xlabel("theta", fontsize=12)
    ax.set_ylabel("epsilon", fontsize=12)
    ax.set_zlabel("L/H consumption ratio", fontsize=12)

    plt.tight_layout()

    plt.savefig(f"{outdir}/lh_ratio_surface_colored.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{outdir}/lh_ratio_surface_colored.pdf", dpi=600, bbox_inches="tight")

def run_experiment(p_high=0.75, p_low=0.5, tag="baseline"):
    vhigh0, vlow0 = run_learning(p_high=p_high, p_low=p_low, steps=100, theta=3.0, epsilon=0.05)
    print(f"Pre-learning for ({p_high}, {p_low}) -> V_H0={vhigh0:.4f}, V_L0={vlow0:.4f}")

    thetas = [0.05, 0.10, 0.25, 0.5, 1.0, 3.0, 4.0]
    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_reps = 100

    args = [
        (theta, epsilon, seed, p_high, p_low, vhigh0, vlow0)
        for theta in thetas
        for epsilon in epsilons
        for seed in range(n_reps)
    ]

    print(f"Running {len(args)} simulations for p_high={p_high}, p_low={p_low}...")

    with Pool() as pool:
        results = pool.map(run_single_sim, args)

    df = pd.DataFrame(
        results,
        columns=["theta", "epsilon", "delta_v", "Value_High", "Value_Low", "LH_Ratio"]
    )


    df_mean = df.groupby(["theta", "epsilon"], as_index=False).mean()

    outdir = f"resub/interventions/{RUN_TIMESTAMP}/{tag}"
    os.makedirs(outdir, exist_ok=True)

    with open(f"{outdir}/prelearning_init.txt", "w") as f:
        f.write(f"p_high={p_high}, p_low={p_low}, theta_learning=3.0, epsilon_learning=0.05, steps=100\n")
        f.write(f"V_H0={vhigh0:.6f}, V_L0={vlow0:.6f}\n")

    df.to_csv(f"{outdir}/all_agent_results.csv", index=False)
    df_mean.to_csv(f"{outdir}/deltaV_contour_data.csv", index=False)

    plot_lh_ratio_3d(df_mean, outdir)

    print(f"Saved results to {outdir}")
    return df_mean

def plot_combined(df1, df2, p1, p2):
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
        ax.set_xlabel(r'$\theta$', fontsize=12)
        ax.tick_params(labelsize=11)
        if ax is axes[0]:
            ax.set_ylabel(r'$\epsilon$', fontsize=12)

        ax.contour(T, G, Z, levels=[0], colors='red', linewidths=1.2)

    cbar = fig.colorbar(
        cs,
        ax=axes,
        orientation='vertical',
        shrink=0.85,
        pad=0.04,
        ticks=[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]
    )
    cbar.set_label(r'$\Delta V = V_H - V_L$', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    combined_dir = f"resub/interventions/{RUN_TIMESTAMP}/contour/"
    os.makedirs(combined_dir, exist_ok=True)

    plt.savefig(f"{combined_dir}/interventions.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{combined_dir}/interventions.pdf", dpi=600, bbox_inches='tight')

if __name__ == "__main__":
    df1 = run_experiment(p_high=0.75, p_low=0.5, tag="baseline")
    df2 = run_experiment(p_high=1.0, p_low=0.5, tag="strong_diff")

    plot_combined(df1, df2, (0.75, 0.5), (1.0, 0.5))