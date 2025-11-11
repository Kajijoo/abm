import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from vector_model import run_vectorized_learning, run_vectorized_simulation

# ===================== CONFIG =====================
EPSILON = 0.05
STEPS = 100
N = 100
WIDTH, HEIGHT = 100, 100

THETA_PRE = 3.0
THETA_POST_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

P_LOW = 0.5
P_HIGH_LIST = np.linspace(0.50, 2.0, 20)

N_REPS = 100

# ===================== Worker =====================
def worker_sim(args):
    theta_post, p_high, vH0, vL0, rep = args
    res = run_vectorized_simulation(
        theta=theta_post, epsilon=EPSILON,
        p_high=p_high, p_low=P_LOW,
        vhigh0=vH0, vlow0=vL0,
        steps=STEPS, N=N, width=WIDTH, height=HEIGHT
    )
    return (p_high, theta_post, res["delta_V"], res["Value_High"], res["Value_Low"])

# ===================== Core run (pure) =====================
def run_lockin_surface():
    # 1) Pre-learning once per p_high
    prelearn = {}
    for p_high in P_HIGH_LIST:
        vH0, vL0 = run_vectorized_learning(
            p_high=p_high, p_low=P_LOW, theta=THETA_PRE,
            epsilon=EPSILON, steps=STEPS, N=N, width=WIDTH, height=HEIGHT
        )
        prelearn[p_high] = (vH0, vL0)

    # 2) Build tasks
    tasks = []
    for p_high in P_HIGH_LIST:
        vH0, vL0 = prelearn[p_high]
        for theta_post in THETA_POST_LIST:
            for rep in range(N_REPS):
                tasks.append((theta_post, p_high, vH0, vL0, rep))

    # 3) Run one pool over all tasks
    procs = min(max(1, cpu_count() - 1), 12) 
    chunksize = max(1, len(tasks) // (procs * 8))

    with Pool(processes=procs) as pool:
        out = pool.map(worker_sim, tasks, chunksize=chunksize)

    # 4) Aggregate to means per (ratio, theta_post)
    rows = {}
    for p_high, theta_post, dV, vH, vL in out:
        ratio = round(p_high / P_LOW, 2)
        key = (ratio, theta_post)
        if key not in rows:
            rows[key] = {"sum_dV":0.0, "sum_vH":0.0, "sum_vL":0.0, "n":0}
        rows[key]["sum_dV"] += dV
        rows[key]["sum_vH"] += vH
        rows[key]["sum_vL"] += vL
        rows[key]["n"] += 1

    results = []
    for (ratio, theta_post), acc in rows.items():
        n = acc["n"]
        results.append({
            "ratio": ratio,
            "theta_post": theta_post,
            "DeltaV_post": acc["sum_dV"]/n,
            "Value_High_post": acc["sum_vH"]/n,
            "Value_Low_post": acc["sum_vL"]/n
        })

    df = pd.DataFrame(results)
    return df

# ===================== Plot =====================
def plot_lockin_surface(df, outdir):
    ratios = sorted(df["ratio"].unique())
    thetas = sorted(df["theta_post"].unique())

    Z = df.pivot(index="theta_post", columns="ratio", values="DeltaV_post").values
    R, T = np.meshgrid(ratios, thetas)

    plt.figure(figsize=(7,5))
    levels = np.linspace(Z.min(), Z.max(), 30)
    cs = plt.contourf(R, T, Z, levels=levels, cmap=plt.cm.viridis)
    plt.contour(R, T, Z, levels=[0], colors='red', linewidths=1.2)

    plt.xlabel("Reward contrast (pH / pL)", fontsize=12)
    plt.ylabel(r'$\theta$', fontsize=12)

    cbar = plt.colorbar(cs, ticks=[-0.5, 0, 0.5, 1.0, 1.5, 2.0])
    cbar.set_label(r'$V_H - V_L$', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "lockin_surface_contour.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(outdir, "lockin_surface_contour.pdf"), dpi=600, bbox_inches="tight")
    plt.show()

# ===================== Main =====================
if __name__ == "__main__":
    t0 = time.time()
    RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTDIR = os.path.join("resub", "lockin", RUN_TIMESTAMP)
    os.makedirs(OUTDIR, exist_ok=True)

    print("Running lock-in surface analysis...")
    df = run_lockin_surface()
    df.to_csv(os.path.join(OUTDIR, "lockin_surface_data.csv"), index=False)
    plot_lockin_surface(df, OUTDIR)

    print(f"\nSaved results to {OUTDIR}")
    print(f"Total runtime: {time.time() - t0:.2f} s")
