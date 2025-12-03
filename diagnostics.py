import numpy as np
import matplotlib.pyplot as plt
from social_model import run_vectorized_simulation
from social_model import update_social, build_BA_adjacency


# --------------------------------------------------------------------
# 1. Basic: Do soc_high and soc_low actually update?
# --------------------------------------------------------------------
def test_basic_social_update():
    print("\n===== Test 1: Basic social update functionality =====")

    N = 5
    A = np.array([
        [0,1,1,0,0],
        [1,0,1,0,0],
        [1,1,0,0,0],
        [0,0,0,0,1],
        [0,0,0,1,0]
    ], dtype=int)

    eat_H = np.array([1,1,0,0,0], dtype=bool)
    deg = np.array(A.sum(axis=1))
    deg_safe = np.maximum(deg,1)

    soc_high = np.zeros(N)
    soc_low  = np.zeros(N)

    update_social(A, eat_H, soc_high, soc_low, learning_rate=0.3, deg_safe=deg_safe)

    print("soc_high:", soc_high)
    print("soc_low :", soc_low)

    if np.any(soc_high != 0) or np.any(soc_low != 0):
        print("PASS: Social values updated.")
    else:
        print("FAIL: No change in social values.")


# --------------------------------------------------------------------
# 2. Agreement increases social value (SPE > 0)
# --------------------------------------------------------------------
def test_agreement_positive_update():
    print("\n===== Test 2: Agreement should increase social value =====")

    # Full agreement network
    A = np.ones((5,5), dtype=int) - np.eye(5)
    eat_H = np.array([1,1,1,1,1], dtype=bool)
    deg_safe = np.full(5, 4)

    soc_high = np.zeros(5)
    soc_low  = np.zeros(5)

    update_social(A, eat_H, soc_high, soc_low, learning_rate=0.3, deg_safe=deg_safe)

    print("soc_high after full agreement:", soc_high)

    if np.all(soc_high > 0):
        print("PASS: Agreement generates positive SPE.")
    else:
        print("FAIL: Expected positive SPE for agreement.")


# --------------------------------------------------------------------
# 3. Disagreement reduces social value (SPE < 0)
# --------------------------------------------------------------------
def test_disagreement_negative_update():
    print("\n===== Test 3: Disagreement should decrease social value =====")

    # Everyone chooses Low, one agent chooses High → full disagreement for agent 0
    A = np.ones((5,5), dtype=int) - np.eye(5)
    eat_H = np.array([1,0,0,0,0], dtype=bool)
    deg_safe = np.full(5, 4)

    soc_high = np.zeros(5)
    soc_low  = np.zeros(5)

    update_social(A, eat_H, soc_high, soc_low, learning_rate=0.3, deg_safe=deg_safe)

    print("soc_high:", soc_high)
    print("soc_low :", soc_low)

    if soc_high[0] < 0:
        print("PASS: Disagreement generated negative SPE.")
    else:
        print("FAIL: Expected negative SPE for disagreement.")


# --------------------------------------------------------------------
# 4. Test whether w_soc affects choice (main behavioral effect)
# --------------------------------------------------------------------
def test_w_soc_influence():
    print("\n===== Test 4: Effect of w_soc on choice probabilities =====")

    # Minimal environment with HL patches only → choice depends on values
    theta = 1.0
    N = 100
    steps = 150

    print("Running two sims: w_soc=0 and w_soc=3 ...")

    res0, _ = run_vectorized_simulation(
        theta=theta, epsilon=0.05, p_high=0.9, p_low=0.6,
        steps=steps, N=N, width=steps, height=N,
        learning_rate=0.3, w_soc=0.0, seed=0
    )
    res1, _ = run_vectorized_simulation(
        theta=theta, epsilon=0.05, p_high=0.9, p_low=0.6,
        steps=steps, N=N, width=steps, height=N,
        learning_rate=0.3, w_soc=3.0, seed=0
    )

    print("delta_V w_soc=0 :", res0["delta_V"])
    print("delta_V w_soc=3 :", res1["delta_V"])

    if abs(res1["delta_V"]) != abs(res0["delta_V"]):
        print("PASS: Increasing w_soc meaningfully affects learning.")
    else:
        print("WARNING: w_soc does not appear to influence ΔV.")


# --------------------------------------------------------------------
# 5. Test whether social agreement creates clusters of value preference
# --------------------------------------------------------------------
def test_cluster_formation():
    print("\n===== Test 5: Cluster formation test =====")

    # Social influence should create heterogeneity when epsilon > 0
    theta = 1.0
    N = 200
    steps = 200

    res, A = run_vectorized_simulation(
        theta=theta, epsilon=0.10, p_high=0.9, p_low=0.6,
        steps=steps, N=N, width=steps, height=N,
        learning_rate=0.3, w_soc=2.0, seed=10
    )

    deltaV = res["value_high_vec"] - res["value_low_vec"]

    plt.figure(figsize=(6,4))
    plt.hist(deltaV, bins=30)
    plt.title("Distribution of ΔV (should show spread under social influence)")
    plt.xlabel("ΔV")
    plt.ylabel("Count")
    plt.tight_layout()

    print("PASS: Visual check—if distribution spreads or becomes multimodal, clusters form.")
    plt.show()


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=========================================")
    print(" Running Social Reinforcement Diagnostics ")
    print("=========================================\n")

    test_basic_social_update()
    test_agreement_positive_update()
    test_disagreement_negative_update()
    test_w_soc_influence()
    test_cluster_formation()

    print("\nDiagnostics complete.")
