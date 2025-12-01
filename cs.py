import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Pool
import os, warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from social_model import run_vectorized_simulation

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['figure.dpi'] = 150


# ================================================================
# Single simulation wrapper (used in multiprocessing)
# ================================================================
def run_single_social_convergence(args):
    w_soc, seed = args
    np.random.seed(seed)

    steps = 200
    N = 100

    res = run_vectorized_simulation(
        theta=1.5,
        epsilon=0.0,
        p_high=0.9,
        p_low=0.6,
        vhigh0=0.01,
        vlow0=0.01,
        steps=steps,
        N=N,
        width=steps,
        height=N,
        seed=seed,
        learning_rate=0.3,
        extinction_rate=1.0,
        w_soc=w_soc,
        delta=0.0,
        record_history=True,
    )

    delta_v_series = np.asarray(res["deltaV_hist"])
    vol = delta_v_series[20:].std()

    return w_soc, delta_v_series, vol


# ================================================================
# Experiment runner
# ================================================================
def run_social_convergence_experiment():

    w_soc_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_reps = 1000

    args = [(w, seed) for w in w_soc_values for seed in range(n_reps)]
    print(f"Running {len(args)} simulations ({len(w_soc_values)} w_soc × {n_reps} reps)...")

    with Pool() as pool:
        results = pool.map(run_single_social_convergence, args)

    records = []
    vol_records = []

    for w_soc, delta_series, vol in results:
        for t, val in enumerate(delta_series):
            records.append((w_soc, t, val))
        vol_records.append((w_soc, vol))

    df = pd.DataFrame(records, columns=["w_soc", "step", "delta_v"])
    df_mean = df.groupby(["w_soc", "step"], as_index=False)["delta_v"].mean()

    df_vol = pd.DataFrame(vol_records, columns=["w_soc", "volatility"])
    df_vol_mean = df_vol.groupby("w_soc", as_index=False)["volatility"].mean()

    os.makedirs("resub/social_convergence", exist_ok=True)
    df_mean.to_csv("resub/social_convergence/convergence_wsoc_mean.csv", index=False)
    df_vol_mean.to_csv("resub/social_convergence/convergence_wsoc_volatility.csv", index=False)

    print("Saved results to resub/social_convergence/")

    return df_mean, df_vol_mean


# ================================================================
# Plotting
# ================================================================
def plot_social_convergence(df_mean, df_vol):

    cmap = plt.cm.viridis
    colors = {
        0.0: cmap(0.0),
        0.25: cmap(0.25),
        0.5: cmap(0.5),
        0.75: cmap(0.75),
        1.0: cmap(1.0)
    }

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))

    # ---- Convergence ----
    ax = axes[0]
    for w_soc, group in df_mean.groupby("w_soc"):
        ax.plot(
            group["step"],
            group["delta_v"],
            lw=1.8,
            color=colors[w_soc],
            label=fr'$w_s$={w_soc}'
        )

    ax.set_xlabel('Steps')
    ax.set_ylabel(r'$V_H - V_L$')
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Convergence", fontsize=11)
    ax.legend(frameon=False, fontsize=9)

    # ---- Volatility ----
    ax2 = axes[1]
    ax2.plot(
        df_vol["w_soc"], df_vol["volatility"],
        marker='o', lw=1.8, color=cmap(0.55)
    )
    ax2.set_xlabel(r'$w_s$')
    ax2.set_ylabel(r'SD of $V_H - V_L$')
    ax2.tick_params(labelsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title("Volatility", fontsize=11)

    plt.tight_layout()
    plt.savefig("resub/social_convergence/social_convergence_plots.png", dpi=600, bbox_inches="tight")
    plt.savefig("resub/social_convergence/social_convergence_plots.pdf", dpi=600, bbox_inches="tight")
    plt.show()


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    df_mean, df_vol = run_social_convergence_experiment()
    plot_social_convergence(df_mean, df_vol)