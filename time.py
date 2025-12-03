import numpy as np
import matplotlib.pyplot as plt
from social_model import run_vectorized_simulation, build_BA_adjacency


def run_intervention_timeseries():

    # -----------------------
    # PARAMETERS
    # -----------------------
    N = 100
    theta_learning = 0.5
    theta_shock = 3.0
    steps_learning = 150
    steps_test = 200
    eps_learning = 0.05
    eps_test = 0.05

    w_values = [0.0, 0.5, 1.0, 2.0, 3.0]

    seed = 0

    # -----------------------
    # PHASE 1 — LEARNING (healthy env)
    # -----------------------
    rng = np.random.default_rng(seed)
    A_learning = build_BA_adjacency(N, m=3, rng=rng)

    resL, _ = run_vectorized_simulation(
        theta=theta_learning,
        epsilon=eps_learning,
        p_high=0.9,
        p_low=0.6,
        steps=steps_learning,
        N=N,
        width=steps_learning,
        height=N,
        learning_rate=0.3,
        w_soc=1.0,
        seed=seed,
        A_override=A_learning,
        record_history=True
    )

    vhigh0 = resL["value_high_vec"]
    vlow0  = resL["value_low_vec"]
    socH0  = resL["soc_high_vec"]
    socL0  = resL["soc_low_vec"]

    # -----------------------
    # PHASE 2 — SHOCK + TEST
    # -----------------------
    plt.figure(figsize=(8,5))

    for w in w_values:

        resT, _ = run_vectorized_simulation(
            theta=theta_shock,
            epsilon=eps_test,
            p_high=0.9,
            p_low=0.6,
            steps=steps_test,
            N=N,
            width=steps_test,
            height=N,
            learning_rate=0.3,
            w_soc=w,
            seed=seed+1,
            A_override=A_learning,
            vhigh0=vhigh0,
            vlow0=vlow0,
            soc_high0=socH0,
            soc_low0=socL0,
            record_history=True
        )

        plt.plot(
            resT["deltaV_hist"],
            label=f"w_soc = {w}"
        )

    plt.axhline(0, color="black", linewidth=0.7)
    plt.xlabel("Time (post-shock)")
    plt.ylabel("Mean ΔV")
    plt.title("Resilience Time Series After Shock (Healthy → Unhealthy)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_intervention_timeseries()
