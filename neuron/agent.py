from mesa import Agent
from enum import Enum
import random


class Behavior(Enum):
    """A neuron can be in one of four states; send, receive or integrate a signal."""

    DORMANT = 0
    SENDING = 1
    RECEIVING = 2 
    RELAYING = 3

class Neuron(Agent):
    def __init__(self, model, initial_behavior):
        super().__init__(model)
        self.behavior = initial_behavior

    def signal(self):
        neighbor_nodes = self.model.grid_neighborhood(self.pos, include_center = False)

        sending_neighbors = [agent for agent in self.model.grid.get_cell_lists_contents(neighbor_nodes)]




class Neuron(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.behavior = None
        self.received_from = None

    def choose(self):
        # Compute probabilities for 'send', 'receive', and 'integrate'
        probabilities = self.compute_behavior_probabilities()
        self.behavior = random.choices(['send', 'receive', 'integrate'], weights=probabilities, k=1)[0]

    def compute_behavior_probabilities(self):
        weights = self.model.weight_matrix[self.unique_id]
        total_weight = sum(weights)
        num_agents = len(self.model.schedule.agents)

        # Probabilities based on weights, normalized
        send_prob = sum(weights) / (total_weight + 1)  # Avoid division by zero
        receive_prob = 1 / num_agents
        integrate_prob = 1 - send_prob - receive_prob

        return [max(0, send_prob), max(0, receive_prob), max(0, integrate_prob)]

    def act(self):
        if self.behavior == 'send':
            self.send()
        elif self.behavior == 'receive':
            self.receive()
        else:
            self.integrate()

    def send(self):
        return

    def receive(self):
        
        self.received_from = None
        return

    def integrate(self):
        # Randomly choose an agent to relay the signal to
        potential_recipients = [agent for agent in self.model.schedule.agents if agent != self]
        recipient = random.choice(potential_recipients)
        recipient.receive()
        recipient.received_from = self.unique_id

    def step(self):
        self.choose()
        self.act()