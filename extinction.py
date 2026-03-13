import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Pool
from datetime import datetime
import warnings, os

warnings.simplefilter(action='ignore', category=FutureWarning)

from vector_model import run_vectorized_simulation

RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['figure.dpi'] = 150


def _contour_crossings(df, value_col, x_col="theta", y_col="gamma", level=0.0):
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

    with open(path, "w") as f:
        f.write("Red contour definition: V_H - V_L = 0\n\n")

        if delta_v_crossings:
            c_df = pd.DataFrame(delta_v_crossings, columns=["gamma", "theta_cross", "kind"])
            f.write("Contour summary (interpolated where needed):\n")
            f.write(f"- Minimum required P(H) = {c_df['theta_cross'].min():.6f}\n")
            f.write(f"- Maximum required P(H) = {c_df['theta_cross'].max():.6f}\n")
            f.write(f"- Maximum gamma on contour = {c_df['gamma'].max():.6f}\n")
            for target_gamma in (0.0, 1.0):
                at_target = c_df[np.isclose(c_df["gamma"], target_gamma, atol=1e-12)]
                if at_target.empty:
                    f.write(f"- Gamma {target_gamma:.1f}: no contour crossing in sampled P(H) range.\n")
                else:
                    f.write(
                        f"- Gamma {target_gamma:.1f}: minimum P(H) = {at_target['theta_cross'].min():.6f}, "
                        f"maximum P(H) = {at_target['theta_cross'].max():.6f}\n"
                    )
        else:
            f.write("Contour status:\n")
            f.write("- No V_H - V_L = 0 crossing in this run.\n")

        shifted = df_mean[df_mean["delta_v"] < 0.0]
        f.write("\nSampled region with learning shifted toward low-reward food (V_H - V_L < 0):\n")
        if shifted.empty:
            f.write("- None in sampled grid.\n")
        else:
            f.write(f"- P(H) range = {shifted['theta'].min():.2f} to {shifted['theta'].max():.2f}\n")
            f.write(f"- Gamma range = {shifted['gamma'].min():.2f} to {shifted['gamma'].max():.2f}\n")

# ---------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------
def run_single_sim(args):
    theta, gamma, seed, p_high, p_low = args
    np.random.seed(seed)

    res = run_vectorized_simulation(
        theta=theta,
        epsilon=0.0,
        p_high=p_high,
        p_low=p_low,
        steps=100,
        N=100,
        width=100,
        height=100,
        extinction_rate=gamma,
        seed=seed
    )
    return theta, gamma, res["delta_V"]


# ---------------------------------------------------------------------
# Batch experiment
# ---------------------------------------------------------------------
def run_experiment(p_high=0.75, p_low=0.5, tag="baseline"):
    thetas = np.linspace(0,1,21)
    gammas = np.linspace(0,1,21)
    n_reps = 20

    args = [
        (theta, gamma, seed, p_high, p_low)
        for theta in thetas
        for gamma in gammas
        for seed in range(n_reps)
    ]

    print(f"Running {len(args)} simulations for p_high={p_high}, p_low={p_low}...")

    with Pool() as pool:
        results = pool.map(run_single_sim, args)

    df = pd.DataFrame(results, columns=["theta", "gamma", "delta_v"])
    df_mean = df.groupby(["theta", "gamma"], as_index=False)["delta_v"].mean()

    outdir = f"resub/theta/{RUN_TIMESTAMP}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    df_mean.to_csv(f"{outdir}/deltaV_contour_data.csv", index=False)
    write_numbers_summary(df_mean, outdir)
    print(f"Saved results to {outdir}")
    return df_mean


# ---------------------------------------------------------------------
# Combined contour plot
# ---------------------------------------------------------------------
def plot_combined(df1, df2, p1, p2):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True, constrained_layout=True)

    plots = [
        (axes[0], df1, p1, '150%'),
        (axes[1], df2, p2, '200%')
    ]

    cmap = plt.cm.coolwarm
    vmin_all, vmax_all = -0.5, 1.0
    norm_all = mpl.colors.TwoSlopeNorm(vmin=vmin_all, vcenter=0.0, vmax=vmax_all)
    levels = np.linspace(vmin_all, vmax_all, 21)

    for ax, df, (p_high, p_low), title in plots:
        thetas = np.sort(df["theta"].unique())
        gammas = np.sort(df["gamma"].unique())
        Z = df.pivot(index="gamma", columns="theta", values="delta_v").values
        T, G = np.meshgrid(thetas, gammas)

        cs = ax.contourf(T, G, Z, levels=levels, cmap=cmap, norm=norm_all)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r'$P(H)$', fontsize=12)
        ax.tick_params(labelsize=11)

        if ax is axes[0]:
            ax.set_ylabel(r'$\gamma$', fontsize=12)

        ax.contour(T, G, Z, levels=[0], colors='red', linewidths=1.2)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Shared colorbar
    cbar = fig.colorbar(
        cs,
        ax=axes,
        orientation='vertical',
        shrink=0.85,
        pad=0.04,
        ticks=[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]
    )
    cbar.set_label(r'$V_H - V_L$', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    outdir = f"resub/theta/{RUN_TIMESTAMP}"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/combined_contours.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{outdir}/combined_contours.pdf", dpi=600, bbox_inches='tight')
    plt.show()


def summarize_extinction_widening(df, label):
    gmin = float(df["gamma"].min())
    gmax = float(df["gamma"].max())

    low = df[np.isclose(df["gamma"], gmin)][["theta", "delta_v"]].rename(
        columns={"delta_v": "delta_v_gamma_min"}
    )
    high = df[np.isclose(df["gamma"], gmax)][["theta", "delta_v"]].rename(
        columns={"delta_v": "delta_v_gamma_max"}
    )

    merged = low.merge(high, on="theta", how="inner")
    merged["widening"] = merged["delta_v_gamma_max"] - merged["delta_v_gamma_min"]
    merged["condition"] = label
    merged["gamma_min"] = gmin
    merged["gamma_max"] = gmax
    return merged


def plot_widening(merged1, merged2):
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.0), constrained_layout=True)
    coolwarm = plt.cm.coolwarm
    line_colors = [coolwarm(0.15), coolwarm(0.85)]

    gmin = merged1["gamma_min"].iloc[0]
    gmax = merged1["gamma_max"].iloc[0]
    ax.plot(
        merged1["theta"],
        merged1["widening"],
        marker="o",
        linewidth=1.4,
        markersize=3.0,
        color=line_colors[0],
        label="150%"
    )
    ax.plot(
        merged2["theta"],
        merged2["widening"],
        marker="s",
        linewidth=1.4,
        markersize=3.0,
        color=line_colors[1],
        label="200%"
    )

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r'$P(H)$', fontsize=12)
    ax.set_ylabel(rf'$(V_H - V_L)\vert_{{\gamma={gmax:.0f}}} - (V_H - V_L)\vert_{{\gamma={gmin:.0f}}}$', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=10)

    outdir = f"resub/theta/{RUN_TIMESTAMP}"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/extinction_widening.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{outdir}/extinction_widening.pdf", dpi=600, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    df1 = run_experiment(p_high=0.75, p_low=0.5, tag="baseline")
    df2 = run_experiment(p_high=1.0, p_low=0.5, tag="strong_diff")
    plot_combined(df1, df2, (0.75, 0.5), (1.0, 0.5))

    widening1 = summarize_extinction_widening(df1, "150%")
    widening2 = summarize_extinction_widening(df2, "200%")
    plot_widening(widening1, widening2)

    outdir = f"resub/theta/{RUN_TIMESTAMP}"
    widening1.to_csv(f"{outdir}/extinction_widening_150.csv", index=False)
    widening2.to_csv(f"{outdir}/extinction_widening_200.csv", index=False)
