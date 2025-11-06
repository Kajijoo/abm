import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from mesa import Model
from mesa.space import MultiGrid, PropertyLayer
from mesa.datacollection import DataCollector
import random
from agent import LearningAgent
import numpy as np


class LearningModel(Model):
    def __init__(self, N = 100, width = 100, height = 100, learning_model='RWE', distribute_patches = 'random', 
                 seed = None, epsilon = 0.05, theta = 1.5, value_low = 0.001, value_high = 0.001, p_low = 0.6, p_high = 0.9):
        super().__init__(seed = seed)
        self.N = N
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, True)
        self.learning_model = learning_model
        self.epsilon = epsilon #amount of noise in eating decision making (or diet)
        self.theta = theta #determines ratio high to low processed foods
        self.value_high = value_high
        self.value_low = value_low
        self.running = True

        self.datacollector = DataCollector(
            model_reporters={
                "Value_Low": lambda m: np.mean([a.value_low for a in m.agents]),
                "Value_High": lambda m: np.mean([a.value_high for a in m.agents])
            },
            agent_reporters={
                "Value_Low": "value_low",
                "Value_High": "value_high",
                "H_count": lambda a: a.foods_consumed["H"],
                "L_count": lambda a: a.foods_consumed["L"]
            }
        )

        if seed is not None:
            random.seed(seed)
            self.random.seed(seed)

        # Add a patch layer for the food types, now using integers
        # 0 = LL, 1 = HL, 2 = HH
        patch_layer = PropertyLayer("patch_type", width = self.grid.width, height = self.grid.height, default_value=-1, dtype=int)
        self.grid.add_property_layer(patch_layer)

        # Distribute patches according to chosen strategy
        if distribute_patches == 'random':
            total_patches = 2 * self.grid.width * self.grid.height
            total_h = int(total_patches * (self.theta / (1 + self.theta)))
            total_l = total_patches - total_h

            patch_list = ['H'] * total_h + ['L'] * total_l
            random.shuffle(patch_list)

            for x in range(self.grid.width):
                for y in range(self.grid.height):
                    if len(patch_list) < 2:
                        break
                    first_patch = patch_list.pop()
                    second_patch = patch_list.pop()
                    if first_patch == 'H' and second_patch == 'H':
                        patch_type = 2  # HH
                    elif first_patch == 'L' and second_patch == 'L':
                        patch_type = 0  # LL
                    else:
                        patch_type = 1  # HL
                    patch_layer.set_cell((x, y), patch_type)

        elif distribute_patches == 'gradient_h':
            for y in range(self.grid.height):
                for x in range(self.grid.width):
                    prob_hh = x / self.grid.width
                    patch_type = 1 if random.random() < prob_hh else 2  # 1=HL, 2=HH
                    patch_layer.set_cell((x, y), patch_type)

        elif distribute_patches == 'gradient_l':
            for y in range(self.grid.height):
                for x in range(self.grid.width):
                    prob_ll = x / self.grid.width
                    patch_type = 1 if random.random() < prob_ll else 0  # 1=HL, 0=LL
                    patch_layer.set_cell((x, y), patch_type)

        elif distribute_patches == 'weekday':
            for x in range(self.grid.width):
                for y in range(self.grid.height):
                    if (x % 7) < 4:
                        patch_type = 0  # LL = Weekdays (Monday-Thursday)
                    elif (x % 7) == 4:
                        patch_type = 1  # HL = Friday
                    else:
                        patch_type = 2  # HH = Weekend (Saturday-Sunday)
                    patch_layer.set_cell((x, y), patch_type)

        # Place agents on grid 
        for i in range(self.N):
            agent = LearningAgent(self, row=i, learning_model=learning_model, epsilon=epsilon, value_low=value_low, value_high=value_high, p_low=p_low, p_high=p_high)
            self.grid.place_agent(agent, (0, i % self.grid.height))

    def step(self):
        self.datacollector.collect(self)
        self.agents.do("step")
        #self.agents.do("advance")