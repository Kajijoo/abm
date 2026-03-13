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
    p_high_pre,
    p_high_post,
    pre_steps,
    post_steps,
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
        p_high=p_high_pre,
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
        p_high=p_high_post,
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


def make_plot(
    df,
    pre_steps,
    theta_pre,
    theta_post,
    epsilon_pre,
    epsilon_post,
    lambda_h_pre,
    lambda_h_post,
    outdir,
):
    fig, axes = plt.subplots(1, 3, figsize=(8.7, 3.5), sharey=True, constrained_layout=True)
    cmap = plt.cm.coolwarm
    color_map = {0.0: cmap(0), 1.0: cmap(0.9)}

    panel_specs = [
        ("theta_intervention", axes[0], rf"Prevalence $P(H)$", fr"$P(H)$: {theta_pre:.1f} $\rightarrow$ {theta_post:.1f}"),
        ("epsilon_intervention", axes[1], rf"Dieting $\varepsilon$", fr"$\varepsilon$: {epsilon_pre:.1f} $\rightarrow$ {epsilon_post:.1f}"),
        ("lambda_h_intervention", axes[2], rf"Food reward $\lambda_H$", fr"$\lambda_H$: {lambda_h_pre:.2f} $\rightarrow$ {lambda_h_post:.2f}"),
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


def main():
    theta_pre = 0.50
    theta_post = 0.20
    theta_const = 0.50
    epsilon_pre = 0.0
    epsilon_post = 0.20
    epsilon_const = 0.0
    lambda_h_pre = 0.75
    lambda_h_post = 0.65
    pre_steps = 75
    post_steps = 75
    p_high = 0.75
    p_low = 0.5
    learning_rate = 0.3
    delta = 0.0
    n_agents = 100
    width = 100
    height = 100
    seed = 123
    outdir_root = "resub/extinction_intervention"

    outdir = os.path.join(outdir_root, RUN_TIMESTAMP)
    os.makedirs(outdir, exist_ok=True)

    frames = []
    for gamma in (0.0, 1.0):
        df_theta = run_two_phase_trajectory(
            gamma=gamma,
            theta_pre=theta_pre,
            theta_post=theta_post,
            epsilon_pre=epsilon_const,
            epsilon_post=epsilon_const,
            p_high_pre=p_high,
            p_high_post=p_high,
            pre_steps=pre_steps,
            post_steps=post_steps,
            p_low=p_low,
            learning_rate=learning_rate,
            delta=delta,
            n_agents=n_agents,
            width=width,
            height=height,
            seed=seed + int(gamma) * 1000,
        )
        df_theta["scenario"] = "theta_intervention"
        frames.append(df_theta)

        df_epsilon = run_two_phase_trajectory(
            gamma=gamma,
            theta_pre=theta_const,
            theta_post=theta_const,
            epsilon_pre=epsilon_pre,
            epsilon_post=epsilon_post,
            p_high_pre=p_high,
            p_high_post=p_high,
            pre_steps=pre_steps,
            post_steps=post_steps,
            p_low=p_low,
            learning_rate=learning_rate,
            delta=delta,
            n_agents=n_agents,
            width=width,
            height=height,
            seed=seed + int(gamma) * 1000 + 100,
        )
        df_epsilon["scenario"] = "epsilon_intervention"
        frames.append(df_epsilon)
        df_lambda_h = run_two_phase_trajectory(
            gamma=gamma,
            theta_pre=theta_const,
            theta_post=theta_const,
            epsilon_pre=epsilon_const,
            epsilon_post=epsilon_const,
            p_high_pre=lambda_h_pre,
            p_high_post=lambda_h_post,
            pre_steps=pre_steps,
            post_steps=post_steps,
            p_low=p_low,
            learning_rate=learning_rate,
            delta=delta,
            n_agents=n_agents,
            width=width,
            height=height,
            seed=seed + int(gamma) * 1000 + 200,
        )
        df_lambda_h["scenario"] = "lambda_h_intervention"
        frames.append(df_lambda_h)

    df = pd.concat(frames, ignore_index=True)
    csv_path = os.path.join(outdir, "extinction_intervention_trajectory.csv")
    df.to_csv(csv_path, index=False)

    make_plot(
        df=df,
        pre_steps=pre_steps,
        theta_pre=theta_pre,
        theta_post=theta_post,
        epsilon_pre=epsilon_pre,
        epsilon_post=epsilon_post,
        lambda_h_pre=lambda_h_pre,
        lambda_h_post=lambda_h_post,
        outdir=outdir,
    )

    print(f"Saved outputs to: {outdir}")
    print(f"- trajectory data: {csv_path}")
    print("- figures: extinction_intervention_trajectory.png / .pdf")


if __name__ == "__main__":
    main()
