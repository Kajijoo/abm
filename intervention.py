import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
from multiprocessing import Pool
import time
import os, warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from vector_model import run_vectorized_learning, run_vectorized_simulation

mpl.rcParams['font.family'] = 'Arial'

# Timestamped root folder
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ---------------------------------------------------------------------
# Learning phase
# ---------------------------------------------------------------------
def run_learning(p_high, p_low, steps=100, theta=0.759, epsilon=0.0, N=100, seed=123):
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


def _contour_crossings(df, value_col, x_col="theta", y_col="epsilon", level=0.0):
    """Interpolate contour crossings as x-values for each fixed y-value."""
    crossings = []
    for y_val, grp in df.groupby(y_col):
        grp = grp.sort_values(x_col)
        x = grp[x_col].to_numpy()
        values = grp[value_col].to_numpy()
        shifted = values - level

        exact = x[np.isclose(shifted, 0.0, atol=1e-12)]
        for x0 in exact:
            crossings.append((float(y_val), float(x0), "exact"))

        for i in range(len(x) - 1):
            x1, x2 = x[i], x[i + 1]
            y1, y2 = shifted[i], shifted[i + 1]
            if y1 == 0.0 or y2 == 0.0:
                continue
            if y1 * y2 < 0.0:
                x0 = x1 - y1 * (x2 - x1) / (y2 - y1)
                crossings.append((float(y_val), float(x0), "interpolated"))

    if not crossings:
        return []

    crossings.sort(key=lambda row: (row[0], row[1]))
    deduped = []
    for row in crossings:
        if not deduped or (
            abs(deduped[-1][0] - row[0]) > 1e-12
            or abs(deduped[-1][1] - row[1]) > 1e-12
        ):
            deduped.append(row)
    return deduped


def write_numbers_summary(df_mean, outdir):
    path = os.path.join(outdir, "numbers_summary.txt")
    delta_v_crossings = _contour_crossings(df_mean, "delta_v", level=0.0)
    df_lh = df_mean.copy()
    df_lh["lh_share"] = np.clip(df_lh["LH_Ratio"] / (1.0 + df_lh["LH_Ratio"]), 0.0, 1.0)
    lh_share_crossings = _contour_crossings(df_lh, "lh_share", level=0.5)

    with open(path, "w") as f:
        f.write("Red contour definition: V_H - V_L = 0\n\n")

        if delta_v_crossings:
            c_df = pd.DataFrame(delta_v_crossings, columns=["epsilon", "theta_cross", "kind"])
            f.write("Contour summary (interpolated where needed):\n")
            f.write(f"- Minimum epsilon needed = {c_df['epsilon'].min():.6f}\n")
            f.write(f"- Minimum required P(H) = {c_df['theta_cross'].min():.6f}\n")
            f.write(f"- Maximum required P(H) = {c_df['theta_cross'].max():.6f}\n")
            f.write(f"- Maximum epsilon on contour = {c_df['epsilon'].max():.6f}\n")
        else:
            f.write("Contour status:\n")
            f.write("- No V_H - V_L = 0 crossing in this run.\n")

        shifted = df_mean[df_mean["delta_v"] < 0.0]
        f.write("\nSampled region with learning shifted toward low-reward food (V_H - V_L < 0):\n")
        if shifted.empty:
            f.write("- None in sampled grid.\n")
        else:
            f.write(f"- P(H) range = {shifted['theta'].min():.2f} to {shifted['theta'].max():.2f}\n")
            f.write(f"- Epsilon range = {shifted['epsilon'].min():.2f} to {shifted['epsilon'].max():.2f}\n")

        f.write("\nRed contour definition: L/(L+H) consumption = 0.5\n")
        if lh_share_crossings:
            c_lh = pd.DataFrame(lh_share_crossings, columns=["epsilon", "theta_cross", "kind"])
            for target_eps in (0.0, 1.0):
                at_target = c_lh[np.isclose(c_lh["epsilon"], target_eps, atol=1e-12)]
                if at_target.empty:
                    f.write(f"- Epsilon {target_eps:.1f}: no 0.5 contour crossing in sampled P(H) range.\n")
                else:
                    f.write(
                        f"- Epsilon {target_eps:.1f}: minimum P(H) = {at_target['theta_cross'].min():.6f}, "
                        f"maximum P(H) = {at_target['theta_cross'].max():.6f}\n"
                    )
        else:
            f.write("- No L/(L+H) = 0.5 contour crossing in this run.\n")


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def _lh_surface(df):
    thetas = np.sort(df["theta"].unique())
    eps = np.sort(df["epsilon"].unique())
    z_ratio = df.pivot(index="epsilon", columns="theta", values="LH_Ratio").values
    z_share = z_ratio / (1.0 + z_ratio)
    z_share = np.clip(z_share, 0.0, 1.0)
    t_grid, e_grid = np.meshgrid(thetas, eps)
    return t_grid, e_grid, z_share


def plot_lh_ratio_heatmap(df, outdir):
    T, E, Z = _lh_surface(df)

    fig, ax = plt.subplots(figsize=(4.0, 3.2), constrained_layout=True)
    levels = np.linspace(0.0, 1.0, 21)
    cs = ax.contourf(T, E, Z, levels=levels, cmap="coolwarm_r", vmin=0.0, vmax=1.0)
    ax.contour(T, E, Z, levels=[0.5], colors="red", linewidths=1.2)

    ax.set_xlabel(r'$P(H)$', fontsize=12)
    ax.set_ylabel(r'$\epsilon$', fontsize=12)
    ax.tick_params(labelsize=11)

    cbar = fig.colorbar(cs, ax=ax, shrink=0.9, pad=0.02, ticks=np.linspace(0.0, 1.0, 6))
    cbar.set_label(r'$L/(L+H)$ consumption', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(f"{outdir}/lh_ratio_heatmap.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{outdir}/lh_ratio_heatmap.pdf", dpi=600, bbox_inches="tight")


def plot_lh_ratio_combined(df1, df2, root_outdir):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True, constrained_layout=True)
    plots = [(axes[0], df1, "150%"), (axes[1], df2, "200%")]
    levels = np.linspace(0.0, 1.0, 21)

    for ax, df, title in plots:
        T, E, Z = _lh_surface(df)
        cs = ax.contourf(T, E, Z, levels=levels, cmap="coolwarm_r", vmin=0.0, vmax=1.0)
        ax.contour(T, E, Z, levels=[0.5], colors="red", linewidths=1.2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r'$P(H)$', fontsize=12)
        ax.tick_params(labelsize=11)
        xticks = np.linspace(0.0, 1.0, 6)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1f}" for x in xticks])
        if ax is axes[0]:
            ax.set_ylabel(r'$\epsilon$', fontsize=12)

    cbar = fig.colorbar(cs, ax=axes, orientation='vertical', shrink=0.85,
                        pad=0.04, ticks=np.linspace(0.0, 1.0, 6))
    cbar.set_label(r'$L/(L+H)$ consumption', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    plt.savefig(f"{root_outdir}/lh_ratio_combined.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{root_outdir}/lh_ratio_combined.pdf", dpi=600, bbox_inches='tight')


def plot_combined(df1, df2, p1, p2):
    os.makedirs(f"resub/interventions/{RUN_TIMESTAMP}", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True, constrained_layout=True)

    plots = [
        (axes[0], df1, p1, '150%'),
        (axes[1], df2, p2, '200%')
    ]

    cmap = plt.cm.coolwarm
    vmin_all, vmax_all = -0.25, 1.0
    norm_all = mpl.colors.TwoSlopeNorm(vmin=vmin_all, vcenter=0.0, vmax=vmax_all)
    levels = np.linspace(vmin_all, vmax_all, 21)

    for ax, df, (p_high, p_low), title in plots:
        thetas = np.sort(df["theta"].unique())
        eps = np.sort(df["epsilon"].unique())
        Z = df.pivot(index="epsilon", columns="theta", values="delta_v").values
        T, G = np.meshgrid(thetas, eps)

        cs = ax.contourf(T, G, Z, levels=levels, cmap=cmap, norm=norm_all)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r'$P(H)$', fontsize=12)
        ax.tick_params(labelsize=11)
        xticks = np.linspace(0.0, 1.0, 6)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1f}" for x in xticks])

        if ax is axes[0]:
            ax.set_ylabel(r'$\epsilon$', fontsize=12)

        ax.contour(T, G, Z, levels=[0], colors='red', linewidths=1.2)

    # Shared colorbar
    cbar = fig.colorbar(cs, ax=axes, orientation='vertical', shrink=0.85,
                        pad=0.04, ticks=[-0.25, 0, 0.25, 0.5, 0.75, 1.0])
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
    vhigh0, vlow0 = run_learning(p_high, p_low, steps=100, theta=0.759, epsilon=0)
    print(f"Pre-learning ({p_high}, {p_low}) -> V_H0={vhigh0:.4f}, V_L0={vlow0:.4f}")

    thetas = np.linspace(0, 1, 21)
    epsilons = np.linspace(0, 1, 21)
    n_reps = 20

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
        f.write(f"p_high={p_high}, p_low={p_low}")
        f.write(f"V_H0={vhigh0:.6f}, V_L0={vlow0:.6f}\n")

    df.to_csv(f"{outdir}/all_agent_results.csv", index=False)
    df_mean.to_csv(f"{outdir}/deltaV_contour_data.csv", index=False)
    write_numbers_summary(df_mean, outdir)

    plot_lh_ratio_heatmap(df_mean, outdir)
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
    plot_lh_ratio_combined(df1, df2, f"resub/interventions/{RUN_TIMESTAMP}")

    total_time = time.time() - start_time
    print(f"\n=== Total runtime: {total_time/60:.2f} minutes ===")
