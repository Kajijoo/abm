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
    thetas = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.80, 1.0, 2.0, 4.0]
    gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_reps = 100

    args = [(theta, gamma, seed, p_high, p_low)
            for theta in thetas
            for gamma in gammas
            for seed in range(n_reps)]

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

    cmap = plt.cm.viridis
    vmin_all, vmax_all = -0.5, 1.0
    norm_all = mpl.colors.Normalize(vmin=vmin_all, vmax=vmax_all)
    levels = np.linspace(vmin_all, vmax_all, 21)

    for ax, df, (p_high, p_low), title in plots:
        thetas = np.sort(df["theta"].unique())
        gammas = np.sort(df["gamma"].unique())
        Z = df.pivot(index="gamma", columns="theta", values="delta_v").values
        T, G = np.meshgrid(thetas, gammas)

        cs = ax.contourf(T, G, Z, levels=levels, cmap=cmap, norm=norm_all)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r'$\theta$', fontsize=12)
        ax.tick_params(labelsize=11)

        if ax is axes[0]:
            ax.set_ylabel(r'$\gamma$', fontsize=12)

        # Add red ΔV=0 contour line
        contour = ax.contour(T, G, Z, levels=[0], colors='red', linewidths=1.2)

        # Add x-tick at lowest γ where ΔV=0 occurs
        if len(contour.allsegs[0]) > 0:
            segs = contour.allsegs[0]
            lowest_g = G.min()
            segs_lowest = [s for s in segs if np.any(np.isclose(s[:,1], lowest_g))]
            if len(segs_lowest) == 0:
                segs_lowest = [segs[0]]
            x_cross = np.concatenate([s[:,0] for s in segs_lowest])
            if len(x_cross) > 0:
                x_cross_val = float(np.round(np.mean(x_cross), 2))

                # keep integer ticks except 0, add transition tick
                base_ticks = [int(t) for t in ax.get_xticks() if t.is_integer() and t != 0]
                all_ticks = sorted(set(base_ticks + [x_cross_val]))
                ax.set_xticks(all_ticks)

                # format tick labels
                labels = [
                    f"{t:.2f}" if abs(t - x_cross_val) < 1e-3 else f"{int(t)}"
                    for t in all_ticks
                ]
                ax.set_xticklabels(labels)

    # Shared colorbar
    cbar = fig.colorbar(cs, ax=axes, orientation='vertical',
                        shrink=0.85, pad=0.04,
                        ticks=[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label(r'$V_H - V_L$', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    outdir = f"resub/theta/{RUN_TIMESTAMP}"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/combined_contours.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{outdir}/combined_contours.pdf", dpi=600, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    df1 = run_experiment(p_high=0.75, p_low=0.5, tag="baseline")
    df2 = run_experiment(p_high=1.0, p_low=0.5, tag="strong_diff")
    plot_combined(df1, df2, (0.75, 0.5), (1.0, 0.5))
