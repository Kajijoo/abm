import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import os, time

from social_model import run_vectorized_simulation, build_BA_adjacency


# ============================================================
# PHASE 1 – Learning Phase
# ============================================================

def run_learning_phase(seed, theta_learning, epsilon, w_soc, N):

    # Build network ONCE
    rng = np.random.default_rng(seed)
    A = build_BA_adjacency(N, m=1, rng=rng)

    # Run learning simulation
    res, _ = run_vectorized_simulation(
        theta=theta_learning,
        p_high=0.9,
        p_low=0.6,
        steps=150,
        N=N,
        width=150,
        height=N,
        w_soc=w_soc,
        epsilon=epsilon,
        learning_rate=0.3,
        vhigh0=None,
        vlow0=None,
        seed=seed,
        A_override=A,          
        soc_high0=None,         
        soc_low0=None
    )

    return (
        res["value_high_vec"],
        res["value_low_vec"],
        res["soc_high_vec"],
        res["soc_low_vec"],
        A
    )


# ============================================================
# PHASE 2 – Intervention Phase
# ============================================================

def run_test_phase(args):
    epsilon, w_soc, seed, vhigh0, vlow0, soc_high0, soc_low0, A = args

    res, _ = run_vectorized_simulation(
        theta=3,
        p_high=0.9,
        p_low=0.6,
        steps=150,
        N=len(vhigh0),
        width=150,
        height=len(vhigh0),
        w_soc=w_soc,
        epsilon=epsilon,
        learning_rate=0.3,
        vhigh0=vhigh0,
        vlow0=vlow0,
        soc_high0=soc_high0,   
        soc_low0=soc_low0,
        seed=seed,
        A_override=A         
    )

    vh = res["value_high_vec"]
    vl = res["value_low_vec"]

    deltaV = vh - vl

    return epsilon, w_soc, deltaV.mean(), deltaV.var()


# ============================================================
# MAIN
# ============================================================

def run_intervention_experiment():

    N = 100
    theta_learning = 0.25

    print("Running learning phase...")
    (
        vhigh0_vec,
        vlow0_vec,
        soc_high0_vec,
        soc_low0_vec,
        A_learning
    ) = run_learning_phase(
        seed=0,
        theta_learning=theta_learning,
        epsilon=0.05,
        w_soc=1.0,
        N=N
    )


    # ---------------------------------------------------------
    # TEST PHASE SWEEP
    # ---------------------------------------------------------

    epsilons = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00]
    w_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    n_reps = 20

    args = [
        (eps, w, seed, vhigh0_vec, vlow0_vec,
         soc_high0_vec, soc_low0_vec, A_learning)
        for eps in epsilons
        for w in w_values
        for seed in range(n_reps)
    ]

    print(f"Running {len(args)} simulations in test phase...")

    with mp.Pool() as pool:
        results = pool.map(run_test_phase, args)

    df = pd.DataFrame(results, columns=["epsilon", "w_soc", "mean_deltaV", "var_deltaV"])
    df_mean = df.groupby(["epsilon", "w_soc"]).mean().reset_index()

    outdir = "resub/social_intervention_corrected"
    os.makedirs(outdir, exist_ok=True)

    df.to_csv(f"{outdir}/all_results.csv", index=False)
    df_mean.to_csv(f"{outdir}/summary.csv", index=False)

    print("Saved results.")


    # ---------------------------------------------------------
    # PLOT
    # ---------------------------------------------------------

    pivot = df_mean.pivot(index="w_soc", columns="epsilon", values="mean_deltaV")
    EPS = pivot.columns.values
    W = pivot.index.values
    Z = pivot.values

    fig, ax = plt.subplots(figsize=(7,5))
    cs = ax.contourf(EPS, W, Z, levels=20, cmap="viridis")
    plt.colorbar(cs, label="Mean ΔV (lower = more resilient)")

    ax.set_xlabel("epsilon")
    ax.set_ylabel("w_soc")
    ax.set_title("Resilience After Healthy Learning")

    plt.tight_layout()
    plt.savefig(f"{outdir}/resilience_contour.png", dpi=300)
    plt.savefig(f"{outdir}/resilience_contour.pdf", dpi=300)

    return df_mean



if __name__ == "__main__":
    start = time.time()
    run_intervention_experiment()
    print(f"Done. Runtime {(time.time()-start)/60:.2f} min")