import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Pool

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from model import LearningModel

# Global plot style
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['figure.dpi'] = 150


def run_single_convergence(args):
    """Run a LearningModel simulation and return ΔV_t over time for all agents."""
    alpha, seed = args
    np.random.seed(seed)

    steps = 200
    N = 100
    theta = 1.5
    gamma = 0.3

    model = LearningModel(N=N, width=steps, height=N, learning_model='RWE', theta=theta, epsilon=0)

    for a in model.agents:
        a.learning_rate = alpha
        a.extinction_rate = gamma
        a.value_high = 0.9
        a.value_low = 0.6

    # Track ΔV_t for all agents (for volatility calc)
    agent_deltaV = np.zeros((N, steps))

    for t in range(steps):
        model.step()
        for i, a in enumerate(model.agents):
            agent_deltaV[i, t] = a.value_high - a.value_low

    # Compute population mean over agents at each step
    delta_v_series = agent_deltaV.mean(axis=0)
    # Compute per-agent volatility (std across time, skipping first 20 steps)
    per_agent_vol = agent_deltaV[:, 20:].std(axis=1).mean()

    return alpha, delta_v_series, per_agent_vol


def run_convergence_experiment():
    """Run convergence simulations across learning rates."""
    alphas = [0.1, 0.3, 0.6, 1.0]
    n_reps = 100
    args = [(alpha, seed) for alpha in alphas for seed in range(n_reps)]

    print(f"Running {len(args)} simulations ({len(alphas)} α levels × {n_reps} seeds)...")

    with Pool() as pool:
        results = pool.map(run_single_convergence, args)

    # aggregate results
    records = []
    vol_records = []

    for alpha, delta_v_series, vol in results:
        for t, val in enumerate(delta_v_series):
            records.append((alpha, t, val))
        vol_records.append((alpha, vol))

    df = pd.DataFrame(records, columns=["alpha", "step", "delta_v"])
    df_mean = df.groupby(["alpha", "step"], as_index=False)["delta_v"].mean()

    df_vol = pd.DataFrame(vol_records, columns=["alpha", "volatility"])
    df_vol_mean = df_vol.groupby("alpha", as_index=False)["volatility"].mean()

    df_mean.to_csv("resub/convergence/convergence_alpha_mean.csv", index=False)
    df_vol_mean.to_csv("resub/convergence/convergence_alpha_volatility.csv", index=False)
    print("Saved mean and volatility results to 'resub/convergence/'")

    return df_mean, df_vol_mean


def plot_convergence_and_volatility(df_mean, df_vol):
    """Plot ΔV_t convergence and volatility across α values."""
    cmap = plt.cm.viridis
    colors = {
        0.1: cmap(0.15),
        0.3: cmap(0.4),
        0.6: cmap(0.65),
        1.0: cmap(0.9)
    }

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), sharey=False)

    # --- Left panel: mean convergence ---
    ax = axes[0]
    for alpha, group in df_mean.groupby("alpha"):
        ax.plot(group["step"], group["delta_v"], lw=1.8, color=colors[alpha],
                label=fr'$\alpha$={alpha}')

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

    # --- Right panel: per-agent volatility vs α ---
    ax2 = axes[1]
    ax2.plot(df_vol["alpha"], df_vol["volatility"], marker='o', lw=1.8, color=colors[0.3])

    ax2.set_xlabel(r'$\alpha$', fontsize=12)
    ax2.set_ylabel(r'SD of $V_H - V_L$', fontsize=12)
    ax2.tick_params(labelsize=10)
    ax2.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('Volatility', fontsize=12)

    plt.tight_layout()
    plt.savefig("resub/convergence/convergence_volatility.png", dpi=600, bbox_inches='tight')
    plt.savefig("resub/convergence/convergence_volatility.pdf", dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    df_mean, df_vol = run_convergence_experiment()
    plot_convergence_and_volatility(df_mean, df_vol)
