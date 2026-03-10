import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Pool
from datetime import datetime
import warnings, os

warnings.simplefilter(action='ignore', category=FutureWarning)

from vector_model import run_vectorized_simulation

RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['figure.dpi'] = 150

# ---------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------
def run_single_sim(args):
    theta, gamma, seed, p_high, p_low = args
    np.random.seed(seed)

    res = run_vectorized_simulation(
        theta=theta,
        epsilon=0.0,
        p_high=p_high,
        p_low=p_low,
        steps=100,
        N=100,
        width=100,
        height=100,
        extinction_rate=gamma,
        seed=seed
    )
    return theta, gamma, res["delta_V"]


# ---------------------------------------------------------------------
# Batch experiment
# ---------------------------------------------------------------------
def run_experiment(p_high=0.75, p_low=0.5, tag="baseline"):
    thetas = np.linspace(0,1,21)
    gammas = np.linspace(0,1,21)
    n_reps = 20

    args = [
        (theta, gamma, seed, p_high, p_low)
        for theta in thetas
        for gamma in gammas
        for seed in range(n_reps)
    ]

    print(f"Running {len(args)} simulations for p_high={p_high}, p_low={p_low}...")

    with Pool() as pool:
        results = pool.map(run_single_sim, args)

    df = pd.DataFrame(results, columns=["theta", "gamma", "delta_v"])
    df_mean = df.groupby(["theta", "gamma"], as_index=False)["delta_v"].mean()

    outdir = f"resub/theta/{RUN_TIMESTAMP}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    df_mean.to_csv(f"{outdir}/deltaV_contour_data.csv", index=False)
    print(f"Saved results to {outdir}")
    return df_mean


# ---------------------------------------------------------------------
# Combined contour plot
# ---------------------------------------------------------------------
def plot_combined(df1, df2, p1, p2):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True, constrained_layout=True)

    plots = [
        (axes[0], df1, p1, '150%'),
        (axes[1], df2, p2, '200%')
    ]

    cmap = plt.cm.coolwarm
    vmin_all, vmax_all = -0.5, 1.0
    norm_all = mpl.colors.TwoSlopeNorm(vmin=vmin_all, vcenter=0.0, vmax=vmax_all)
    levels = np.linspace(vmin_all, vmax_all, 21)

    for ax, df, (p_high, p_low), title in plots:
        thetas = np.sort(df["theta"].unique())
        gammas = np.sort(df["gamma"].unique())
        Z = df.pivot(index="gamma", columns="theta", values="delta_v").values
        T, G = np.meshgrid(thetas, gammas)

        cs = ax.contourf(T, G, Z, levels=levels, cmap=cmap, norm=norm_all)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r'$P(H)$', fontsize=12)
        ax.tick_params(labelsize=11)

        if ax is axes[0]:
            ax.set_ylabel(r'$\gamma$', fontsize=12)

        ax.contour(T, G, Z, levels=[0], colors='red', linewidths=1.2)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Shared colorbar
    cbar = fig.colorbar(
        cs,
        ax=axes,
        orientation='vertical',
        shrink=0.85,
        pad=0.04,
        ticks=[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]
    )
    cbar.set_label(r'$V_H - V_L$', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    outdir = f"resub/theta/{RUN_TIMESTAMP}"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/combined_contours.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{outdir}/combined_contours.pdf", dpi=600, bbox_inches='tight')
    plt.show()


def summarize_extinction_widening(df, label):
    gmin = float(df["gamma"].min())
    gmax = float(df["gamma"].max())

    low = df[np.isclose(df["gamma"], gmin)][["theta", "delta_v"]].rename(
        columns={"delta_v": "delta_v_gamma_min"}
    )
    high = df[np.isclose(df["gamma"], gmax)][["theta", "delta_v"]].rename(
        columns={"delta_v": "delta_v_gamma_max"}
    )

    merged = low.merge(high, on="theta", how="inner")
    merged["widening"] = merged["delta_v_gamma_max"] - merged["delta_v_gamma_min"]
    merged["condition"] = label
    merged["gamma_min"] = gmin
    merged["gamma_max"] = gmax
    return merged


def plot_widening(merged1, merged2):
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.0), constrained_layout=True)
    coolwarm = plt.cm.coolwarm
    line_colors = [coolwarm(0.15), coolwarm(0.85)]

    gmin = merged1["gamma_min"].iloc[0]
    gmax = merged1["gamma_max"].iloc[0]
    ax.plot(
        merged1["theta"],
        merged1["widening"],
        marker="o",
        linewidth=1.4,
        markersize=3.0,
        color=line_colors[0],
        label="150%"
    )
    ax.plot(
        merged2["theta"],
        merged2["widening"],
        marker="s",
        linewidth=1.4,
        markersize=3.0,
        color=line_colors[1],
        label="200%"
    )

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r'$P(H)$', fontsize=12)
    ax.set_ylabel(rf'$(V_H - V_L)\vert_{{\gamma={gmax:.0f}}} - (V_H - V_L)\vert_{{\gamma={gmin:.0f}}}$', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=10)

    outdir = f"resub/theta/{RUN_TIMESTAMP}"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/extinction_widening.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{outdir}/extinction_widening.pdf", dpi=600, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    df1 = run_experiment(p_high=0.75, p_low=0.5, tag="baseline")
    df2 = run_experiment(p_high=1.0, p_low=0.5, tag="strong_diff")
    plot_combined(df1, df2, (0.75, 0.5), (1.0, 0.5))

    widening1 = summarize_extinction_widening(df1, "150%")
    widening2 = summarize_extinction_widening(df2, "200%")
    plot_widening(widening1, widening2)

    outdir = f"resub/theta/{RUN_TIMESTAMP}"
    widening1.to_csv(f"{outdir}/extinction_widening_150.csv", index=False)
    widening2.to_csv(f"{outdir}/extinction_widening_200.csv", index=False)
