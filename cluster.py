import matplotlib.pyplot as plt
import networkx as nx

from social_model import run_vectorized_simulation


def run_clustering_test(
    N=100,
    steps=100,
    w_soc=4.0,
    learning_rate=0.3,
    theta=1,
    epsilon=0.05,
    p_high=0.5,
    p_low=0.5,
    seed=0
):
    # Run simulation
    res, A = run_vectorized_simulation(
        N=N,
        steps=steps,
        width=steps,
        height=N,
        w_soc=w_soc,
        learning_rate=learning_rate,
        epsilon=epsilon,
        theta=theta,
        p_high=p_high,
        p_low=p_low,
        vhigh0=0.001,
        vlow0=0.001,
        record_history=False,
        seed=seed
    )


    # Compute deltaV per agent
    value_high = res["value_high_vec"]
    value_low  = res["value_low_vec"]

    deltaV = value_high - value_low

    # Convert adjacency to networkx
    G = nx.from_scipy_sparse_array(A)


    # Layout
    pos = nx.spring_layout(G, seed=0)

    # Plot
    plt.figure(figsize=(6, 6))
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=60,
        node_color=deltaV,
        cmap="viridis"
    )
    nx.draw_networkx_edges(G, pos, alpha=0.25, width=0.5)

    plt.colorbar(nodes, label=fr"$\Delta V = V_h - V_l$")
    plt.title(f"Clustering test, w_soc={w_soc}, steps={steps}")
    plt.axis("off")
    plt.savefig("clustering_test.png", dpi=150)
    plt.show()


# run test
run_clustering_test()