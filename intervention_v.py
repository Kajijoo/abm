import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
from multiprocessing import Pool
import time
import os, warnings
from scipy.interpolate import RegularGridInterpolator
warnings.simplefilter(action='ignore', category=FutureWarning)

from vector_model import run_vectorized_learning, run_vectorized_simulation

mpl.rcParams['font.family'] = 'Arial'

# Timestamped root folder
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ---------------------------------------------------------------------
# Learning phase
# ---------------------------------------------------------------------
def run_learning(p_high, p_low, steps=100, theta=3.0, epsilon=0.0, N=10000, seed=123):
    np.random.seed(seed)
    vH, vL = run_vectorized_learning(
        p_high=p_high, p_low=p_low, theta=theta, epsilon=epsilon,
        steps=steps, N=N, width=100, height=N, seed=seed
    )
    return vH, vL


# ---------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------
def run_single_sim(args):
    theta, epsilon, seed, p_high, p_low, vhigh0, vlow0 = args
    np.random.seed(seed)

    res = run_vectorized_simulation(
        theta=theta,
        epsilon=epsilon,
        p_high=p_high,
        p_low=p_low,
        vhigh0=vhigh0,
        vlow0=vlow0,
        steps=100,
        N=100,
        width=100,
        height=100,
        seed=seed
    )

    return (
        theta,
        epsilon,
        res["delta_V"],
        res["Value_High"],
        res["Value_Low"],
        res["LH_Ratio"]
    )


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def plot_lh_ratio_3d(df, outdir):
    thetas = sorted(df["theta"].unique())
    eps = sorted(df["epsilon"].unique())
    T, E = np.meshgrid(thetas, eps)

    Z_ratio = df.pivot(index="epsilon", columns="theta", values="LH_Ratio").values
    Z_delta = df.pivot(index="epsilon", columns="theta", values="delta_v").values

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection="3d")

    # --- Surface colored by ΔV, slightly transparent ---
    norm = plt.Normalize(Z_delta.min(), Z_delta.max())
    colors = plt.cm.viridis(norm(Z_delta))
    ax.plot_surface(
        T, E, Z_ratio,
        facecolors=colors,
        edgecolor="none",
        alpha=0.85,        # transparency to reveal red line
        zorder=1
    )

    # --- Compute ΔV = 0 contour (row-by-row interpolation) ---
    contour_pts = []
    for i in range(Z_delta.shape[0]):
        row = Z_delta[i, :]
        for j in range(len(row) - 1):
            if (row[j] <= 0 and row[j + 1] >= 0) or (row[j] >= 0 and row[j + 1] <= 0):
                t0, t1 = thetas[j], thetas[j + 1]
                z0, z1 = row[j], row[j + 1]
                frac = abs(z0) / (abs(z0) + abs(z1)) if (abs(z0) + abs(z1)) > 0 else 0
                theta_cross = t0 + frac * (t1 - t0)
                contour_pts.append((eps[i], theta_cross))

    # --- Interpolate z-values from LH ratio grid for contour points ---
    if contour_pts:
        eps_pts, theta_pts = zip(*contour_pts)
        interp_func = RegularGridInterpolator((eps, thetas), Z_ratio, bounds_error=False, fill_value=np.nan)
        z_vals = interp_func(np.array(contour_pts))

        # Plot ΔV=0 line clearly on top of surface
        ax.plot(
            theta_pts,
            eps_pts,
            z_vals,
            color='red',
            linewidth=2,
            zorder=10,       # force drawn on top
            alpha = 0.9
        )

    # --- Colorbar ---
    mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array(Z_delta)
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label=r'$V_H - V_L$', ticks=[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])

    # --- Labels and appearance ---
    ax.set_xlabel(r'$\theta$', fontsize=12)
    ax.set_ylabel(r'$\epsilon$', fontsize=12)
    ax.set_zlabel("L/H consumption ratio", fontsize=12)
    ax.legend(loc='upper left', frameon=False)
    ax.view_init(elev=25, azim=-60)  # optional: a good default view angle

    plt.tight_layout()
    plt.savefig(f"{outdir}/lh_ratio_surface_colored.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{outdir}/lh_ratio_surface_colored.pdf", dpi=600, bbox_inches="tight")




def plot_combined(df1, df2, p1, p2):
    os.makedirs(f"resub/interventions/{RUN_TIMESTAMP}", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True, constrained_layout=True)

    plots = [
        (axes[0], df1, p1, '150%'),
        (axes[1], df2, p2, '200%')
    ]

    cmap = plt.cm.viridis
    vmin_all, vmax_all = -0.5, 1.0
    norm_all = mpl.colors.Normalize(vmin=vmin_all, vmax=vmax_all)
    levels = np.linspace(vmin_all, vmax_all, 21)

    for ax, df, (p_high, p_low), title in plots:
        thetas = np.sort(df["theta"].unique())
        eps = np.sort(df["epsilon"].unique())
        Z = df.pivot(index="epsilon", columns="theta", values="delta_v").values
        T, G = np.meshgrid(thetas, eps)

        cs = ax.contourf(T, G, Z, levels=levels, cmap=cmap, norm=norm_all)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r'$\theta$', fontsize=12)
        ax.tick_params(labelsize=11)

        if ax is axes[0]:
            ax.set_ylabel(r'$\epsilon$', fontsize=12)

        # Add ΔV=0 contour line (red)
        contour = ax.contour(T, G, Z, levels=[0], colors='red', linewidths=1.2)

        # For 150% plot only, find θ-crossing at lowest epsilon
        if title == '150%' and len(contour.allsegs[0]) > 0:
            segs = contour.allsegs[0]
            lowest_eps = G.min()
            segs_lowest = [s for s in segs if np.any(np.isclose(s[:,1], lowest_eps))]
            if len(segs_lowest) == 0:
                segs_lowest = [segs[0]]
            x_cross = np.concatenate([s[:,0] for s in segs_lowest])
            if len(x_cross) > 0:
                x_cross_val = float(np.round(np.mean(x_cross), 2))

                # keep integer ticks except 0, add transition tick
                base_ticks = [int(t) for t in ax.get_xticks() if t.is_integer() and t != 0]
                base_ticks = sorted(set(base_ticks))
                all_ticks = base_ticks + [x_cross_val]
                all_ticks = sorted(all_ticks)
                ax.set_xticks(all_ticks)

                # format tick labels
                labels = [
                    f"{t:.2f}" if abs(t - x_cross_val) < 1e-3 else f"{int(t)}"
                    for t in all_ticks
                ]
                ax.set_xticklabels(labels)

    # Shared colorbar
    cbar = fig.colorbar(cs, ax=axes, orientation='vertical', shrink=0.85,
                        pad=0.04, ticks=[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label(r'$V_H - V_L$', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    plt.savefig(f"resub/interventions/{RUN_TIMESTAMP}/interventions.png",
                dpi=600, bbox_inches='tight')
    plt.savefig(f"resub/interventions/{RUN_TIMESTAMP}/interventions.pdf",
                dpi=600, bbox_inches='tight')

# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------
def run_experiment(p_high=0.75, p_low=0.5, tag="baseline"):
    vhigh0, vlow0 = run_learning(p_high, p_low, steps=100, theta=3.0, epsilon=0)
    print(f"Pre-learning ({p_high}, {p_low}) -> V_H0={vhigh0:.4f}, V_L0={vlow0:.4f}")

    thetas = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.80, 1.0, 2.0, 4.0]
    epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_reps = 100

    args = [(theta, epsilon, seed, p_high, p_low, vhigh0, vlow0)
            for theta in thetas for epsilon in epsilons for seed in range(n_reps)]

    print(f"Running {len(args)} simulations for p_high={p_high}, p_low={p_low}...")
    with Pool() as pool:
        results = pool.map(run_single_sim, args)

    df = pd.DataFrame(results, columns=["theta", "epsilon", "delta_v",
                                        "Value_High", "Value_Low", "LH_Ratio"])
    df_mean = df.groupby(["theta", "epsilon"], as_index=False).mean()

    outdir = f"resub/interventions/{RUN_TIMESTAMP}/{tag}"
    os.makedirs(outdir, exist_ok=True)

    with open(f"{outdir}/prelearning_init.txt", "w") as f:
        f.write(f"p_high={p_high}, p_low={p_low}, theta_learning=4.0, "
                f"epsilon_learning=0.05, steps=100\n")
        f.write(f"V_H0={vhigh0:.6f}, V_L0={vlow0:.6f}\n")

    df.to_csv(f"{outdir}/all_agent_results.csv", index=False)
    df_mean.to_csv(f"{outdir}/deltaV_contour_data.csv", index=False)

    plot_lh_ratio_3d(df_mean, outdir)
    print(f"Saved results to {outdir}")
    return df_mean


# ---------------------------------------------------------------------
# Entry point with total runtime
# ---------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()

    df1 = run_experiment(p_high=0.75, p_low=0.5, tag="baseline")
    df2 = run_experiment(p_high=1.0, p_low=0.5, tag="strong_diff")
    plot_combined(df1, df2, (0.75, 0.5), (1.0, 0.5))

    total_time = time.time() - start_time
    print(f"\n=== Total runtime: {total_time/60:.2f} minutes ===")