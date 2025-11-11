import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Pool
import os, warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from vector_model import run_vectorized_simulation

# Global plot style
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['figure.dpi'] = 150

# ================================================================
# Vectorized convergence simulation
# ================================================================
def run_single_convergence(args):
    alpha, seed = args
    np.random.seed(seed)

    steps = 200
    N = 100

    # run simulation with history
    res = run_vectorized_simulation(
        theta=1.5,
        epsilon=0.0,
        p_high=0.9,
        p_low=0.6,
        vhigh0=0.9,
        vlow0=0.6,
        steps=steps,
        N=N,
        width=steps,
        height=N,
        seed=seed,
        learning_rate=alpha,
        record_history=True
    )

    delta_v_series = res["deltaV_hist"]

    # volatility = std over time after burn-in
    vol = delta_v_series[20:].std()

    return alpha, delta_v_series, vol


# ================================================================
# Experiment runner
# ================================================================
def run_convergence_experiment():
    alphas = [0.1, 0.3, 0.6, 1.0]
    n_reps = 1000
    args = [(alpha, seed) for alpha in alphas for seed in range(n_reps)]

    print(f"Running {len(args)} simulations ({len(alphas)} α values × {n_reps} reps)...")

    with Pool() as pool:
        results = pool.map(run_single_convergence, args)

    # Aggregate
    records = []
    vol_records = []

    for alpha, delta_series, vol in results:
        for t, val in enumerate(delta_series):
            records.append((alpha, t, val))
        vol_records.append((alpha, vol))

    df = pd.DataFrame(records, columns=["alpha", "step", "delta_v"])
    df_mean = df.groupby(["alpha", "step"], as_index=False)["delta_v"].mean()

    df_vol = pd.DataFrame(vol_records, columns=["alpha", "volatility"])
    df_vol_mean = df_vol.groupby("alpha", as_index=False)["volatility"].mean()

    os.makedirs("resub/convergence", exist_ok=True)
    df_mean.to_csv("resub/convergence/convergence_alpha_mean.csv", index=False)
    df_vol_mean.to_csv("resub/convergence/convergence_alpha_volatility.csv", index=False)

    print("Saved mean and volatility results to resub/convergence/")

    return df_mean, df_vol_mean


# ================================================================
# Plotting
# ================================================================
def plot_convergence_and_volatility(df_mean, df_vol):
    cmap = plt.cm.viridis
    colors = {
        0.1: cmap(0.15),
        0.3: cmap(0.4),
        0.6: cmap(0.65),
        1.0: cmap(0.9)
    }

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))

    # ---- Convergence plot ----
    ax = axes[0]
    for alpha, group in df_mean.groupby("alpha"):
        ax.plot(group["step"], group["delta_v"],
                lw=1.8, color=colors[alpha], label=fr'$\alpha$={alpha}')

    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel(r'$V_H - V_L$', fontsize=12)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=9, title="Learning rate")
    ax.set_title("Convergence", fontsize=11)

    # ---- Volatility plot ----
    ax2 = axes[1]
    ax2.plot(df_vol["alpha"], df_vol["volatility"], marker='o',
             lw=1.8, color=cmap(0.55))

    ax2.set_xlabel(r'$\alpha$', fontsize=12)
    ax2.set_ylabel(r'SD of $V_H - V_L$', fontsize=12)
    ax2.tick_params(labelsize=10)
    ax2.set_yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title("Volatility", fontsize=12)

    plt.tight_layout()
    os.makedirs("resub/convergence", exist_ok=True)
    plt.savefig("resub/convergence/convergence_volatility.png", dpi=600, bbox_inches="tight")
    plt.savefig("resub/convergence/convergence_volatility.pdf", dpi=600, bbox_inches="tight")
    plt.show()


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    df_mean, df_vol = run_convergence_experiment()
    plot_convergence_and_volatility(df_mean, df_vol)