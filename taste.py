from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.space import NetworkGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


        # The Learning Agent updates the health-tasty belief using Bayesian formula: P(H|E) = P(E|H) * P(H) / P(E).
        # The healty-tasty belief is about whether people belief that healthy food is tasty or not.
        # P(H|E) is the posterior or the updated healthy-tasty belief based on the observed evidence (encountering various foods).
        # P(E|H) denotes the likelihood of observing evidence E (i.e., observing a healthy-tasty food) if hypothesis H (healthy food is tasty) is true.
        # P(H) denotes the prior belief of a person that healthy food is tasty.
        # P(E) denotes the marginal likelihood which is the probability of observing a healthy-tasty food (E) regardless of whether healthy food is tasty or not.

class LearningAgent(Agent):
    def __init__(self, unique_id, model, row):
        super().__init__(unique_id, model)
        self.row = row
        self.ptype = None  # The current patch type where the agent is located
        self.food_observed = None # Food observed status (healthy-tasty (HT), healthy-not tasty (HN), unhealthy-tasty (UT), and unhealthy-not tasty (UN))

        # P(H) = Prior belief that healthy food is tasty, in this case 50% for everyone. 
        # We could use data from Pivecka et al. (2023) to prior belief and relate it to social class.
        self.ht_belief = 0.5

    def move(self): # Move agent to the right in grid space
        x, y = self.pos
        new_x = (x + 1) % self.model.grid.width
        self.model.grid.move_agent(self, (new_x, y))

    def observe(self):
        # Get current patch type (food)
        self.ptype = self.model.grid.get_cell_list_contents([self.pos])[0].type

        # P(E|H) = Likelihood of observing the evidence E (e.g., observing a healthy-tasty food) if the hypothesis H (healthy food is tasty) is true.
        # How to calibrate P(E|H) in a smart way?
 
        # Observing healthy-tasty food increases belief in healthy food being tasty
        if self.ptype == "HT":
            likelihood_ht = 0.6

        # Observing healthy but not tasty food decreases belief in healthy food being tasty
        elif self.ptype == "HN":
            likelihood_ht = 0.4

        # Observing unhealthy but tasty food decreases belief in healthy food being tasty
        elif self.ptype == "UT":
            likelihood_ht = 0.4

        # Observing unhealthy but not tasty food increases belief in healthy food being tasty
        else:
            likelihood_ht = 0.6

        marginal_ht = self.p_marginal_ht()

        # P(H|E) = P(E|H) * P(H) / P(E).
        self.ht_belief = (likelihood_ht * self.ht_belief) / marginal_ht

    def p_marginal_ht(self):
        # P(E) = Total probability of observing tasty and healthy food, irrespective of whether healthy food is tasty or not.
        # If the focus is not on model comparison, the marginal likelihood is simply the normalizing constant that ensures that the posterior is a proper probability (i.e., between 0 and 1).
        # We could use data from Kunz et al. (2023) or Pivecka et al. (2023) to calculate marginal likelihood.

        return 0.5

    #def social_influence(self):   
    # We could define some type of social network and define a social influence method that also provides evidence for Bayesian Inference Making
    # Or some a simple weighted average of local neighbors
    
    #       G = nx.erdos_renyi_graph(n = self.num_agents, p = 0.01)
    #       self.grid = NetworkGrid(G) 


    def step(self): 
        self.move()
        self.observe()
        #self.social_influence()
        
class Patch(Agent):
    def __init__(self, unique_id, model, patch_type):
        super().__init__(unique_id, model)
        self.type = patch_type

    def get_color(self):
        if self.type == "HT":
            return "green"
        elif self.type == "HN":
            return "blue"
        elif self.type == "UT":
            return "red"
        else:
            return "purple"

class LearningModel(Model):
    def __init__(self, N, width, height, distribute_patches = 'random', HT = 0.25, HN = 0.25, UT = 0.25, UN = 0.25, seed = None):
        super().__init__()
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = SimultaneousActivation(self)
        self.HT = HT
        self.HN = HN
        self.UT = UT
        self.UN = UN
        
        if seed is not None:
            random.seed(seed)
            self.random.seed(seed)

        #Create agents 
        for i in range(self.num_agents):
            agent = LearningAgent(i, self, row=i)
            self.grid.place_agent(agent, (0, i))
            self.schedule.add(agent)

        #Add patches with types based on different distributions
        if distribute_patches == 'random':
            total_patches = self.grid.width * self.grid.height

            #Distribute HT, HN, UT, and UN randomly on grid according based on probability 
            patch_distribution = ['HT'] * int(total_patches * HT) + \
                                 ['HN'] * int(total_patches * HN) + \
                                 ['UT'] * int(total_patches * UT) + \
                                 ['UN'] * int(total_patches * UN)
            random.shuffle(patch_distribution)

            # Assign the patch types randomly across the grid
            for x in range(self.grid.width):
                for y in range(self.grid.height):
                    if len(patch_distribution) == 0:
                        break  # No more patches to assign
                    patch_type = patch_distribution.pop()
                    patch = Patch(f'patch_{x}_{y}', self, patch_type)
                    self.grid.place_agent(patch, (x, y))

        self.datacollector = DataCollector(
            agent_reporters={'Healthy Tasty Belief':'ht_belief'}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def visualize(self):
        grid_matrix = []
        for y in range(self.grid.height):
            row = []
            for x in range(self.grid.width):
                cell_content = self.grid.get_cell_list_contents([(x,y)])
                patch = next((obj for obj in cell_content if isinstance(obj, Patch)), None)
                if patch:
                    row.append(patch.get_color())
                else:
                    row.append("white")
            grid_matrix.append(row)

        fig, ax = plt.subplots(figsize=(10, 10))
        cmap = mcolors.ListedColormap(['red', 'blue', 'purple', 'green'])
        bounds = [0, 1, 2, 3, 4]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        matrix = [[bounds.index(cmap.colors.index(color)) for color in row] for row in grid_matrix]
        ax.imshow(matrix, cmap = cmap, norm = norm)

        plt.show()