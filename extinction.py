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
    """Run one LearningModel simulation for a given theta, gamma, and reward setup."""
    theta, gamma, seed, p_high, p_low = args
    np.random.seed(seed)
    
    steps = 100
    N = 100

    model = LearningModel(N=N, width=steps, height=N,
                          learning_model='RWE', theta=theta, epsilon=0.0)

    for a in model.agents:
        a.learning_rate = 0.3
        a.extinction_rate = gamma
        a.p_high = p_high
        a.p_low = p_low
        a.value_high = 0.001
        a.value_low = 0.001

    for _ in range(steps):
        model.step()

    vals_high = np.mean([a.value_high for a in model.agents])
    vals_low = np.mean([a.value_low for a in model.agents])
    delta_v = vals_high - vals_low

    return theta, gamma, delta_v


def run_experiment(p_high=0.9, p_low=0.6, tag="baseline"):
    """Run full grid of (theta, gamma) for given reward parameters."""
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


def plot_contour(df, p_high, p_low, tag):
    thetas = np.sort(df["theta"].unique())
    gammas = np.sort(df["gamma"].unique())
    Z = df.pivot(index="gamma", columns="theta", values="delta_v").values
    T, G = np.meshgrid(thetas, gammas)

    vmin, vmax = -p_low, p_high
    levels = np.linspace(vmin, vmax, 11)  # fixed edges
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    cmap = plt.cm.viridis

    contourf = ax.contourf(T, G, Z, levels=levels, cmap=cmap, norm=norm, extend='neither')

    cbar = fig.colorbar(
        contourf,
        ax=ax,
        ticks=[p_high, 0, -p_low],   # fixed ticks
        boundaries=levels
    )
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    cbar.set_label(r'$\Delta V = V_H - V_L$', fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    ax.set_xlabel(r'$\theta$', fontsize=12)
    ax.set_ylabel(r'$\gamma$', fontsize=12)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(f"resub/theta/{tag}/contour_deltaV.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"resub/theta/{tag}/contour_deltaV.pdf", dpi=600, bbox_inches='tight')

if __name__ == "__main__":
    # Baseline case (150% reward difference)
    df1 = run_experiment(p_high=0.9, p_low=0.6, tag="baseline")
    plot_contour(df1, p_high=0.9, p_low=0.6, tag="baseline")

    # Stronger reward contrast (200% difference)
    df2 = run_experiment(p_high=1.0, p_low=0.5, tag="strong_diff")
    plot_contour(df2, p_high=1.0, p_low=0.5, tag="strong_diff")