import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Pool
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)

from model import LearningModel

# Global plot style
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['figure.dpi'] = 150


def run_single_sim(args):
    theta, gamma, seed, p_high, p_low = args
    np.random.seed(seed)
    steps = 100
    N = 100

    model = LearningModel(N=N, width=steps, height=N,
                          learning_model='RWE', theta=theta, epsilon=0.0)

    for a in model.agents:
        a.extinction_rate = gamma
        a.p_high = p_high
        a.p_low = p_low

    for _ in range(steps):
        model.step()

    df = model.datacollector.get_model_vars_dataframe()
    last = df.iloc[-1]
    vals_low = float(last["Value_Low"])
    vals_high = float(last["Value_High"])
    delta_v = vals_high - vals_low

    return theta, gamma, delta_v


def run_experiment(p_high=0.75, p_low=0.5, tag="baseline"):
    thetas = np.linspace(0.0, 4.0, 40)
    gammas = np.linspace(0.0, 1.0, 10)
    n_reps = 1

    args = [(theta, gamma, seed, p_high, p_low)
            for theta in thetas
            for gamma in gammas
            for seed in range(n_reps)]

    print(f"Running {len(args)} simulations for p_high={p_high}, p_low={p_low}...")

    with Pool() as pool:
        results = pool.map(run_single_sim, args)

    df = pd.DataFrame(results, columns=["theta", "gamma", "delta_v"])
    df_mean = df.groupby(["theta", "gamma"], as_index=False)["delta_v"].mean()

    outdir = f"resub/theta/{tag}"
    os.makedirs(outdir, exist_ok=True)
    df_mean.to_csv(f"{outdir}/deltaV_contour_data.csv", index=False)
    print(f"Saved results to {outdir}")
    return df_mean


def plot_combined(df1, df2, p1, p2):
    # Use constrained layout for reliable alignment
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

        c = ax.contourf(T, G, Z, levels=levels, cmap=cmap, norm=norm_all)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r'$\theta$', fontsize=12)
        ax.tick_params(labelsize=11)
        if ax is axes[0]:
            ax.set_ylabel(r'$\gamma$', fontsize=12)

        # Red highlight at delta V = 0
        ax.contour(
            T, G, Z,
            levels=[0],
            colors='red',
            linewidths=1.2
        )

    # Shared colorbar on the right
    cbar = fig.colorbar(
        c, ax=axes, orientation='vertical',
        shrink=0.85, pad=0.04,
        ticks=[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]
    )
    cbar.set_label(r'$\Delta V = V_H - V_L$', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    plt.savefig("resub/theta/combined_contours.png", dpi=600, bbox_inches='tight')
    plt.savefig("resub/theta/combined_contours.pdf", dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 150% reward difference
    df1 = run_experiment(p_high=0.75, p_low=0.5, tag="baseline")

    # 200% reward difference
    df2 = run_experiment(p_high=1.0, p_low=0.5, tag="strong_diff")

    plot_combined(df1, df2, (0.75, 0.5), (1.0, 0.5))