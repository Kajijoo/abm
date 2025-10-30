from mesa import Model
from mesa.space import NetworkGrid
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from agent import Neuron, Behavior

class Brain(Model):
    def __init__(self, N = 20, beta = 5, seed = None):
        super().__init__(seed = seed)
        self.num_agents = N
        self.beta = beta
        self.weight_matrix = np.zeros((N, N))
        self.G = self.network()
        self.grid = NetworkGrid(self.G)

        for node in self.G.nodes():
            a = Neuron(self,
                       Behavior.DORMANT
                       )
            self.grid.place_agents(a, node)

        sending_nodes = self.random.sample(list(self.G), self.beta)
        for a in self.grid.get_cell_list_contents(sending_nodes):
            a.behavior = Behavior.SENDING

        receiving_nodes = self.random.sample(list(self.G), self.beta)
        for a in self.grid.get_cell_list_contents(receiving_nodes):
            a.behavior = Behavior.RECEIVING

        relaying_nodes = self.random.sample(list(self.G), self.beta)
        for a in self.grid.get_cell_list_contents(relaying_nodes):
            a.behavior = Behavior.RELAYING


    def update_weights(self, active_agents):
        for agent_i in active_agents:
            for agent_j in active_agents:
                if agent_i != agent_j:
                    self.update_weight(agent_i, agent_j)


    def update_weight(self, agent_i, agent_j):
        """
        Updates the weight matrix based on interactions between agent_i and agent_j.
        """
        updated = False

        # Case 1: Agent_i sends, Agent_j receives
        if agent_i.behavior == 'send' and agent_j.behavior == 'receive':
            self.weight_matrix[agent_i.unique_id, agent_j.unique_id] += 0.1
            self.weight_matrix[agent_j.unique_id, agent_i.unique_id] += 0.1
            print(f"Weight increased: Agent {agent_i.unique_id} -> Agent {agent_j.unique_id}")
            updated = True

        # Case 2: Agent_i sends, Agent_j integrates
        elif agent_i.behavior == 'send' and agent_j.behavior == 'integrate':
            self.weight_matrix[agent_i.unique_id, agent_j.unique_id] += 0.1
            self.weight_matrix[agent_j.unique_id, agent_i.unique_id] += 0.1
            print(f"Weight increased: Agent {agent_i.unique_id} -> Agent {agent_j.unique_id}")
            updated = True

        # Case 3: Both Agent_i and Agent_j send
        elif agent_i.behavior == 'send' and agent_j.behavior == 'send':
            self.weight_matrix[agent_i.unique_id, agent_j.unique_id] -= 0.1
            self.weight_matrix[agent_j.unique_id, agent_i.unique_id] -= 0.1
            print(f"Weight decreased: Agent {agent_i.unique_id} <-> Agent {agent_j.unique_id}")
            updated = True

        # Case 4: Chain reaction (send -> integrate -> receive)
        if agent_j.behavior == 'receive' and agent_j.received_from is not None:
            integrator = self.schedule.agents[agent_j.received_from]
            if integrator.behavior == 'integrate':
                # Strengthen the chain: sender -> integrator -> receiver
                self.weight_matrix[agent_i.unique_id, integrator.unique_id] += 0.1
                self.weight_matrix[integrator.unique_id, agent_j.unique_id] += 0.1
                print(
                    f"Chain reaction: Agent {agent_i.unique_id} -> Integrator {integrator.unique_id} -> Receiver {agent_j.unique_id}"
                )
                updated = True

        # Ensure weights remain non-negative
        self.weight_matrix = np.maximum(self.weight_matrix, 0)

        if not updated:
            print(f"No weight update: Agent {agent_i.unique_id} and Agent {agent_j.unique_id}")


    def step(self):
        active_agents = random.sample(self.schedule.agents, k=random.randint(1, self.beta))
        for agent in active_agents:
            agent.shuffle_do("step")

        self.update_weights(active_agents)

    def network(self):
        G = nx.Graph()

        # Add edges to the graph for non-zero weights
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if self.weight_matrix[i, j] > 0:  # Include only positive weights
                    G.add_edge(i, j, weight=self.weight_matrix[i, j])

        # Check if any edges were added
        if len(G.edges) == 0:
            print("No edges were added to the graph. Check the weight matrix.")

        # Get edge weights for visualization
        edge_data = nx.get_edge_attributes(G, 'weight')
        if not edge_data:
            print("No edge attributes found. Check edge addition.")
            return None

        edges, weights = zip(*edge_data.items())

        # Draw the graph
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_color='lightblue', with_labels=True, node_size=500, font_size=10)
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=[weight for weight in weights])

        plt.show()
        return G