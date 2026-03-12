import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vector_model import run_vectorized_simulation


RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def run_two_phase_trajectory(
    gamma,
    theta_pre,
    theta_post,
    epsilon_pre,
    epsilon_post,
    pre_steps,
    post_steps,
    p_high,
    p_low,
    learning_rate,
    delta,
    n_agents,
    width,
    height,
    seed,
):
    pre = run_vectorized_simulation(
        theta=theta_pre,
        epsilon=epsilon_pre,
        p_high=p_high,
        p_low=p_low,
        steps=pre_steps,
        N=n_agents,
        width=width,
        height=height,
        seed=seed,
        vhigh0=0.0,
        vlow0=0.0,
        learning_rate=learning_rate,
        extinction_rate=gamma,
        delta=delta,
        record_history=True,
    )

    post = run_vectorized_simulation(
        theta=theta_post,
        epsilon=epsilon_post,
        p_high=p_high,
        p_low=p_low,
        steps=post_steps,
        N=n_agents,
        width=width,
        height=height,
        seed=seed + 1,
        vhigh0=pre["Value_High"],
        vlow0=pre["Value_Low"],
        learning_rate=learning_rate,
        extinction_rate=gamma,
        delta=delta,
        record_history=True,
    )

    delta_hist = np.concatenate([pre["deltaV_hist"], post["deltaV_hist"]])
    steps = np.arange(1, pre_steps + post_steps + 1)

    return pd.DataFrame(
        {
            "step": steps,
            "delta_v": delta_hist,
            "gamma": gamma,
        }
    )


def make_plot(df, pre_steps, theta_pre, theta_post, epsilon_pre, epsilon_post, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.5), sharey=True, constrained_layout=True)
    cmap = plt.cm.coolwarm
    color_map = {0.0: cmap(0), 1.0: cmap(0.9)}

    panel_specs = [
        ("theta_intervention", axes[0], r"Food environment", fr"$P(H)$: {theta_pre:.1f} $\rightarrow$ {theta_post:.1f}"),
        ("epsilon_intervention", axes[1], r"Individual", fr"$\epsilon$: {epsilon_pre:.1f} $\rightarrow$ {epsilon_post:.1f}"),
    ]

    for scenario, ax, title, annotation in panel_specs:
        df_panel = df[df["scenario"] == scenario]
        for gamma in (0.0, 1.0):
            sub = df_panel[df_panel["gamma"] == gamma]
            ax.plot(
                sub["step"],
                sub["delta_v"],
                linewidth=2.0,
                color=color_map[gamma],
                label=fr"$\gamma={int(gamma)}$",
            )
        ax.axvline(pre_steps, color="black", linestyle="--", linewidth=2.0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Step")
        ax.text(0.02, 0.96, annotation, transform=ax.transAxes, fontsize=10, va="top")

    axes[0].set_ylabel(r"$V_H - V_L$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=1, frameon=False, bbox_to_anchor=(0.98,0.3))

    png_path = os.path.join(outdir, "extinction_intervention_trajectory.png")
    pdf_path = os.path.join(outdir, "extinction_intervention_trajectory.pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot compact side-by-side V_H - V_L trajectories for gamma=0 and gamma=1: "
            "left panel changes P(H), right panel changes epsilon."
        )
    )
    parser.add_argument("--theta-pre", type=float, default=0.50, help="P(H) before intervention in left panel.")
    parser.add_argument("--theta-post", type=float, default=0.20, help="P(H) after intervention in left panel.")
    parser.add_argument("--theta-const", type=float, default=0.50, help="Constant P(H) in right panel.")
    parser.add_argument("--epsilon-pre", type=float, default=0.0, help="Epsilon before intervention in right panel.")
    parser.add_argument("--epsilon-post", type=float, default=0.20, help="Epsilon after intervention in right panel.")
    parser.add_argument("--epsilon-const", type=float, default=0.0, help="Constant epsilon in left panel.")
    parser.add_argument("--pre-steps", type=int, default=75, help="Steps before intervention.")
    parser.add_argument("--post-steps", type=int, default=75, help="Steps after intervention.")
    parser.add_argument("--p-high", type=float, default=0.75, help="Reward probability for H when consumed.")
    parser.add_argument("--p-low", type=float, default=0.5, help="Reward probability for L when consumed.")
    parser.add_argument("--learning-rate", type=float, default=0.3, help="Learning rate.")
    parser.add_argument("--delta", type=float, default=0.0, help="Learning exponent parameter.")
    parser.add_argument("--n-agents", type=int, default=100, help="Number of agents.")
    parser.add_argument("--width", type=int, default=100, help="Grid width.")
    parser.add_argument("--height", type=int, default=100, help="Grid height.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--outdir-root",
        type=str,
        default="resub/extinction_intervention",
        help="Root output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    outdir = os.path.join(args.outdir_root, RUN_TIMESTAMP)
    os.makedirs(outdir, exist_ok=True)

    frames = []
    for gamma in (0.0, 1.0):
        df_theta = run_two_phase_trajectory(
            gamma=gamma,
            theta_pre=args.theta_pre,
            theta_post=args.theta_post,
            epsilon_pre=args.epsilon_const,
            epsilon_post=args.epsilon_const,
            pre_steps=args.pre_steps,
            post_steps=args.post_steps,
            p_high=args.p_high,
            p_low=args.p_low,
            learning_rate=args.learning_rate,
            delta=args.delta,
            n_agents=args.n_agents,
            width=args.width,
            height=args.height,
            seed=args.seed + int(gamma) * 1000,
        )
        df_theta["scenario"] = "theta_intervention"
        frames.append(df_theta)

        df_epsilon = run_two_phase_trajectory(
            gamma=gamma,
            theta_pre=args.theta_const,
            theta_post=args.theta_const,
            epsilon_pre=args.epsilon_pre,
            epsilon_post=args.epsilon_post,
            pre_steps=args.pre_steps,
            post_steps=args.post_steps,
            p_high=args.p_high,
            p_low=args.p_low,
            learning_rate=args.learning_rate,
            delta=args.delta,
            n_agents=args.n_agents,
            width=args.width,
            height=args.height,
            seed=args.seed + int(gamma) * 1000 + 100,
        )
        df_epsilon["scenario"] = "epsilon_intervention"
        frames.append(df_epsilon)

    df = pd.concat(frames, ignore_index=True)
    csv_path = os.path.join(outdir, "extinction_intervention_trajectory.csv")
    df.to_csv(csv_path, index=False)

    make_plot(
        df=df,
        pre_steps=args.pre_steps,
        theta_pre=args.theta_pre,
        theta_post=args.theta_post,
        epsilon_pre=args.epsilon_pre,
        epsilon_post=args.epsilon_post,
        outdir=outdir,
    )

    print(f"Saved outputs to: {outdir}")
    print(f"- trajectory data: {csv_path}")
    print("- figures: extinction_intervention_trajectory.png / .pdf")


if __name__ == "__main__":
    main()
