import time
from model import LearningModel
from vector_model import run_vectorized_simulation

def run_mesa_model(theta=1.5, epsilon=0.25, p_high=0.9, p_low=0.6, steps=100, N=500):
    model = LearningModel(
        N=N,
        width=100,
        height=100,
        learning_model="RWE",
        theta=theta,
        epsilon=epsilon,
    )
    for a in model.agents:
        a.p_high = p_high
        a.p_low = p_low

    t0 = time.time()
    for _ in range(steps):
        model.step()
    t1 = time.time()

    vhighs = [a.value_high for a in model.agents]
    vlows = [a.value_low for a in model.agents]
    deltas = [vh - vl for vh, vl in zip(vhighs, vlows)]
    l_counts = [a.foods_consumed["L"] for a in model.agents]
    h_counts = [a.foods_consumed["H"] for a in model.agents]
    ratios = [Lc / Hc if Hc > 0 else float("nan") for Lc, Hc in zip(l_counts, h_counts)]

    return {
        "Value_High": sum(vhighs) / N,
        "Value_Low": sum(vlows) / N,
        "delta_V": sum(deltas) / N,
        "LH_Ratio": sum(r for r in ratios if r == r) / max(1, len([r for r in ratios if r == r])),
        "runtime": t1 - t0,
    }


def run_vector_model(theta=1.5, epsilon=0.25, p_high=0.9, p_low=0.6, steps=100, N=500):
    t0 = time.time()
    result = run_vectorized_simulation(
        theta=theta,
        epsilon=epsilon,
        p_high=p_high,
        p_low=p_low,
        steps=steps,
        N=N,
        width=100,
        height=100,
        seed=42,
    )
    t1 = time.time()
    result["runtime"] = t1 - t0
    return result


if __name__ == "__main__":
    params = dict(theta=1.5, epsilon=0.25, p_high=0.9, p_low=0.6, steps=100, N=500)

    print("Running Mesa model...")
    mesa_res = run_mesa_model(**params)
    print("Running vectorized model...")
    vec_res = run_vector_model(**params)

    print("\n=== RESULTS COMPARISON ===")
    for k in ["Value_High", "Value_Low", "delta_V", "LH_Ratio"]:
        print(f"{k:12s} Mesa={mesa_res[k]:.4f} | Vector={vec_res[k]:.4f}")

    print(f"\nRuntime (Mesa): {mesa_res['runtime']:.2f} s")
    print(f"Runtime (Vectorized): {vec_res['runtime']:.2f} s")
    print(f"Speedup: {mesa_res['runtime']/vec_res['runtime']:.1f}x faster")
