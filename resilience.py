import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import time

from social_model import run_vectorized_simulation


# ============================================================
# PARAMETERS
# ============================================================

epsilons = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
w_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0]

n_reps = 30

N = 100
steps = 150

theta = 3
p_high = 0.9
p_low = 0.6
learning_rate = 0.3


# ============================================================
# ONE RUN
# ============================================================

def run_single(args):
    eps, w_soc, seed = args

    res, _A = run_vectorized_simulation(
        N=N,
        steps=steps,
        width=steps,
        height=N,
        w_soc=w_soc,
        learning_rate=learning_rate,
        epsilon=eps,
        theta=theta,
        p_high=p_high,
        p_low=p_low,
        vhigh0=None,
        vlow0=None,
        record_history=False,
        seed=seed,
    )

    vh = res["value_high_vec"]
    vl = res["value_low_vec"]

    deltaV = vh - vl

    mean_deltaV = deltaV.mean()
    var_deltaV = deltaV.var()

    return eps, w_soc, mean_deltaV, var_deltaV


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    RUN_ID = time.strftime("%Y%m%d_%H%M%S")
    outdir = f"resub/epsilon_w/{RUN_ID}"
    os.makedirs(outdir, exist_ok=True)

    # create job list
    args = [(eps, w, seed) for eps in epsilons for w in w_values for seed in range(n_reps)]

    print(f"Running {len(args)} simulations...")

    with mp.Pool() as pool:
        results = pool.map(run_single, args)

    df = pd.DataFrame(
        results,
        columns=["epsilon", "w_soc", "mean_deltaV", "var_deltaV"]
    )

    # aggregate
    df_mean = df.groupby(["epsilon", "w_soc"]).mean().reset_index()

    df_mean.to_csv(f"{outdir}/epsilon_w_results.csv", index=False)

    print(f"Saved CSV to {outdir}/epsilon_w_results.csv")


    # ========================================================
    # CONTOUR PLOT
    # ========================================================

    # pivot for contour
    pivot = df_mean.pivot(index="w_soc", columns="epsilon", values="mean_deltaV")

    EPS = pivot.columns.values
    W = pivot.index.values
    Z = pivot.values

    fig, ax = plt.subplots(figsize=(7,5))

    cs = ax.contourf(EPS, W, Z, levels=20, cmap="viridis")
    cbar = plt.colorbar(cs)
    cbar.set_label("ΔV")

    ax.set_xlabel("epsilon")
    ax.set_ylabel("w_soc")

    plt.tight_layout()
    plt.savefig(f"{outdir}/epsilon_w_contour.png", dpi=300)
    plt.savefig(f"{outdir}/epsilon_w_contour.pdf", dpi=300)

    print(f"Saved plots to {outdir}")