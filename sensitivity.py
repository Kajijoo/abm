import argparse
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vector_model import run_vectorized_learning, run_vectorized_simulation


RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _rank_series(values):
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def _pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom == 0:
        return np.nan
    return float((x * y).sum() / denom)


def _partial_corr_rank(df, x_col, y_col, control_cols):
    x = _rank_series(df[x_col].values)
    y = _rank_series(df[y_col].values)

    if len(control_cols) == 0:
        return _pearson_corr(x, y)

    Z = np.column_stack([_rank_series(df[c].values) for c in control_cols])
    Z = np.column_stack([np.ones(Z.shape[0]), Z])  # intercept

    bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
    by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    rx = x - Z @ bx
    ry = y - Z @ by
    return _pearson_corr(rx, ry)


def _sample_parameters(n_samples, seed, bounds):
    rng = np.random.default_rng(seed)
    rows = []
    for sample_id in range(n_samples):
        rows.append(
            {
                "sample_id": sample_id,
                "theta_post": rng.uniform(*bounds["theta_post"]),
                "epsilon": rng.uniform(*bounds["epsilon"]),
                "p_high": rng.uniform(*bounds["p_high"]),
                "p_low": rng.uniform(*bounds["p_low"]),
                "learning_rate": rng.uniform(*bounds["learning_rate"]),
                "extinction_rate": rng.uniform(*bounds["extinction_rate"]),
            }
        )
    return pd.DataFrame(rows)


def _single_run(task):
    sample_id, rep, params, common = task
    seed = common["base_seed"] + sample_id * 10_000 + rep

    #v_h0, v_l0 = run_vectorized_learning(
    #    p_high=params["p_high"],
    #    p_low=params["p_low"],
    #    theta=common["theta_pre"],
    #    epsilon=common["epsilon_pre"],
    #    steps=common["steps"],
    #    N=common["n_agents"],
    #    width=common["width"],
    #    height=common["height"],
    #    seed=seed,
    #    learning_rate=params["learning_rate"],
    #    extinction_rate=params["extinction_rate"],
    #)

    res = run_vectorized_simulation(
        theta=params["theta_post"],
        epsilon=params["epsilon"],
        p_high=params["p_high"],
        p_low=params["p_low"],
        vhigh0=0,#v_h0,
        vlow0=0,#v_l0,
        steps=common["steps"],
        N=common["n_agents"],
        width=common["width"],
        height=common["height"],
        seed=seed,
        learning_rate=params["learning_rate"],
        extinction_rate=params["extinction_rate"],
    )

    out = {
        "sample_id": sample_id,
        "rep": rep,
        "delta_v": res["delta_V"],
        "Value_High": res["Value_High"],
        "Value_Low": res["Value_Low"],
        "LH_Ratio": res["LH_Ratio"],
    }
    out.update(params)
    return out


def run_sensitivity(
    n_samples=500,
    n_reps=20,
    base_seed=1234,
    steps=100,
    n_agents=100,
    width=100,
    height=100,
    theta_pre=0.759,
    epsilon_pre=0.0,
    n_workers=None,
    outdir_root="resub/sensitivity",
):
    bounds = {
        "theta_post": (0.0, 1.0),
        "epsilon": (0.0, 1.0),
        "p_high": (0.50, 1.00),
        "p_low": (0.50, 0.50),
        "learning_rate": (0.0, 1.0),
        "extinction_rate": (0.0, 1.0),
    }

    params_df = _sample_parameters(n_samples=n_samples, seed=base_seed, bounds=bounds)

    common = {
        "base_seed": base_seed,
        "steps": steps,
        "n_agents": n_agents,
        "width": width,
        "height": height,
        "theta_pre": theta_pre,
        "epsilon_pre": epsilon_pre,
    }

    tasks = []
    for row in params_df.to_dict("records"):
        sid = row["sample_id"]
        params = {k: v for k, v in row.items() if k != "sample_id"}
        for rep in range(n_reps):
            tasks.append((sid, rep, params, common))

    if n_workers is None:
        n_workers = max(1, min(cpu_count() - 1, 12))
    chunksize = max(1, len(tasks) // (n_workers * 8))

    with Pool(processes=n_workers) as pool:
        run_rows = pool.map(_single_run, tasks, chunksize=chunksize)

    run_df = pd.DataFrame(run_rows)
    sample_mean_df = (
        run_df.groupby("sample_id", as_index=False)
        .agg(
            {
                "delta_v": "mean",
                "Value_High": "mean",
                "Value_Low": "mean",
                "LH_Ratio": "mean",
                "theta_post": "first",
                "epsilon": "first",
                "p_high": "first",
                "p_low": "first",
                "learning_rate": "first",
                "extinction_rate": "first",
            }
        )
    )

    params = [
        "theta_post",
        "epsilon",
        "p_high",
        "p_low",
        "learning_rate",
        "extinction_rate",
    ]
    varied_params = [p for p in params if sample_mean_df[p].nunique(dropna=False) > 1]

    sens_rows = []
    for p in params:
        is_varied = p in varied_params
        if not is_varied:
            sens_rows.append(
                {
                    "parameter": p,
                    "varied": False,
                    "spearman_rho": np.nan,
                    "abs_spearman_rho": np.nan,
                    "prcc": np.nan,
                    "abs_prcc": np.nan,
                }
            )
            continue

        other = [x for x in varied_params if x != p]
        spearman = _pearson_corr(_rank_series(sample_mean_df[p]), _rank_series(sample_mean_df["delta_v"]))
        prcc = _partial_corr_rank(sample_mean_df, p, "delta_v", other)
        sens_rows.append(
            {
                "parameter": p,
                "varied": True,
                "spearman_rho": spearman,
                "abs_spearman_rho": abs(spearman) if np.isfinite(spearman) else np.nan,
                "prcc": prcc,
                "abs_prcc": abs(prcc) if np.isfinite(prcc) else np.nan,
            }
        )

    sensitivity_df = (
        pd.DataFrame(sens_rows)
        .sort_values(by="abs_prcc", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    outdir = os.path.join(outdir_root, RUN_TIMESTAMP)
    os.makedirs(outdir, exist_ok=True)

    run_df.to_csv(os.path.join(outdir, "all_run_results.csv"), index=False)
    sample_mean_df.to_csv(os.path.join(outdir, "sample_means.csv"), index=False)
    sensitivity_df.to_csv(os.path.join(outdir, "sensitivity_rankings.csv"), index=False)

    plot_sensitivity_bars(sensitivity_df, outdir)
    save_run_config(
        outdir=outdir,
        n_samples=n_samples,
        n_reps=n_reps,
        base_seed=base_seed,
        steps=steps,
        n_agents=n_agents,
        width=width,
        height=height,
        theta_pre=theta_pre,
        epsilon_pre=epsilon_pre,
        n_workers=n_workers,
        bounds=bounds,
    )

    return outdir, sensitivity_df


def plot_sensitivity_bars(sensitivity_df, outdir):
    label_map = {
        "theta_post": r"$P(H)$",
        "epsilon": r"$\varepsilon$",
        "p_high": r"$\lambda_H$",
        "p_low": r"$\lambda_L$",
        "learning_rate": r"$\alpha\beta$",
        "extinction_rate": r"$\gamma$",
    }

    df_plot = (
        sensitivity_df[
            (sensitivity_df["varied"])
            & (sensitivity_df["parameter"] != "delta")
        ]
        .copy()
        .sort_values("abs_prcc", ascending=True)
    )
    if df_plot.empty:
        return
    df_plot["label"] = df_plot["parameter"].map(label_map).fillna(df_plot["parameter"])

    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.85, len(df_plot)))
    ax.barh(df_plot["label"], df_plot["abs_prcc"], color=colors)
    ax.set_xlabel("|PRCC| with $V_H - V_L$")
    ax.set_ylabel("Parameter")
    ax.set_xlim(0, 1)

    plt.savefig(os.path.join(outdir, "sensitivity_prcc.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(outdir, "sensitivity_prcc.pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_run_config(outdir, **cfg):
    with open(os.path.join(outdir, "run_config.txt"), "w", encoding="utf-8") as f:
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Global sensitivity analysis for deltaV = (V_H - V_L), including extinction."
    )
    parser.add_argument("--samples", type=int, default=500, help="Number of parameter samples.")
    parser.add_argument("--reps", type=int, default=20, help="Replications per sample.")
    parser.add_argument("--seed", type=int, default=1234, help="Base random seed.")
    parser.add_argument("--theta-pre", type=float, default=0.759, help="Pre-learning theta.")
    parser.add_argument("--epsilon-pre", type=float, default=0.0, help="Pre-learning epsilon.")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outdir, ranking = run_sensitivity(
        n_samples=args.samples,
        n_reps=args.reps,
        base_seed=args.seed,    
        theta_pre=args.theta_pre,
        epsilon_pre=args.epsilon_pre,
        n_workers=args.workers,
    )

    print(f"Saved sensitivity results to: {outdir}")
    print("\nTop parameters by |PRCC|:")
    print(ranking[["parameter", "prcc", "spearman_rho"]].head(10).to_string(index=False))