from mesa import Agent, Model
from mesa.space import MultiGrid, PropertyLayer
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

class LearningAgent(Agent):
    def __init__(self, model, row, learning_model, epsilon, value_low, value_high):
        super().__init__(model)
        self.row = row
        self.learning_model = learning_model  # 'RW' or 'TD' or 'RWE"
        self.learning_rate = 0.4 # Rate of learning
        self.extinction_rate = 1.0 # Standard for RW extinction = 1, typically <1
        self.delta = 0.0 # Standard for delta = 0. Standard logstic for delta = 1, S-curve 0 < delta < 1
        self.beta = 1.0 # Responsivity to food in TD learning
        self.value_low = value_low  # Initial value outcome
        self.value_high = value_high # Initial value outcome
        self.food_consumed = None  # Food consumed status ('L' or 'H')
        self.p_low = 0.6  # True reward value of food type L
        self.p_high = 0.9  # True reward value of food type H
        self.epsilon = epsilon

    def move(self): # Move agent to the right in grid space
        x, y = self.pos
        new_x = (x + 1) % self.model.grid.width
        self.model.grid.move_agent(self, (new_x, y))

    def eat(self): #Eat procedure for TDW copied from the Hammond et al. 2012 NetLogo model
        #Get current patch type
        ptype = self.model.grid.properties["patch_type"].data[self.pos]

        # Determine food consumed based on patch type
        # 0 = LL, 1 = HL, 2 = HH
        if ptype == 0:  # LL
            self.food_consumed = "L"
        elif ptype == 2:  # HH
            self.food_consumed = "H"
        elif self.value_low == self.value_high:
            self.food_consumed = random.choice(["H", "L"])
        else:
            if self.value_low > self.value_high:
                self.food_consumed = "L"
            else:
                self.food_consumed = "H"
            if random.random() < self.epsilon:
                self.food_consumed = "L" if self.food_consumed == "H" else "H"

    def rw_e(self):
        # Get the patch type of the current agent's position
        ptype = self.model.grid.properties["patch_type"].data[self.pos]
        if ptype == 1:  # HL patch
            # Apply extinction logic only if the agent is on an "HL" patch
            if self.food_consumed == "L":
                self.value_low = self.value_low + (self.learning_rate * (self.value_low ** self.delta) * (self.p_low - self.value_low))
                self.value_high = self.value_high + (self.learning_rate * (self.value_high ** self.delta) * self.extinction_rate * (0 - self.value_high))
            else:
                self.value_high = self.value_high + (self.learning_rate * (self.value_high ** self.delta) * (self.p_high - self.value_high))
                self.value_low = self.value_low + (self.learning_rate * (self.value_low ** self.delta) * self.extinction_rate * (0 - self.value_low))
        else:
            # No extinction logic applied if not on an "HL" patch
            if self.food_consumed == "L":
                self.value_low = self.value_low + (self.learning_rate * (self.p_low - self.value_low))
            else:
                self.value_high = self.value_high + (self.learning_rate * (self.p_high - self.value_high))

    def td(self):
        if self.food_consumed == 'L':
            self.value_low = (self.value_low + self.learning_rate * (self.beta * self.p_low - self.value_low))
        else:
            self.value_high = (self.value_high + self.learning_rate * (self.beta * self.p_high - self.value_high))

    #RW without extinction
    def rw(self):
        if self.food_consumed == "L":
            self.value_low = self.value_low + (self.learning_rate * (self.value_low ** self.delta) * (self.p_low - self.value_low))
        else:
            self.value_high = self.value_high + (self.learning_rate * (self.value_high ** self.delta) * (self.p_high - self.value_high))

    def step(self): 
        self.move()
        
        #Update learning
        if self.learning_model == 'RWE':
            self.eat()
            self.rw_e()
        elif self.learning_model == 'TD':
            self.eat()
            self.td()
        elif self.learning_model == 'RW':
            self.eat()
            self.rw()


class LearningModel(Model):
    def __init__(self, N = 100, width = 20, height = 20, learning_model='RWE', distribute_patches = 'random', seed = None, epsilon = 0.05, theta = 1.5, value_low = 0.001, value_high = 0.001):
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
            agent_reporters={"Value_Low": "value_low", "Value_High": "value_high"}
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
                    prob_hh = y / self.grid.width
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

        print(self.grid.properties["patch_type"].data)

        # Place agents on grid 
        for i in range(self.N):
            agent = LearningAgent(self, row=i, learning_model=learning_model, epsilon=epsilon, value_low=value_low, value_high=value_high)
            self.grid.place_agent(agent, (0, i % self.grid.height))

    def step(self):
        self.datacollector.collect(self)
        self.agents.do("step")
        self.agents.do("advance")


# Function to visualize the grid
def visualize_grid(model):
    """
    Visualize the model's grid with custom colors for patch types:
    - 2 (HH): Red
    - 1 (HL): Purple
    - 0 (LL): Blue
    """
    # Create a colormap
    patch_colors = {
        2: '#FF0000',  # Red for HH
        1: '#800080',  # Purple for HL
        0: '#0000FF',  # Blue for LL
        -1: '#FFFFFF'  # White for empty/unassigned
    }
    
    # Convert grid data to numeric values for plotting
    grid_data = np.zeros((model.grid.height, model.grid.width, 3))
    
    for y in range(model.grid.height):
        for x in range(model.grid.width):
            patch_type = model.grid.properties["patch_type"].data[(x, y)]
            if patch_type in patch_colors:
                # Convert hex color to RGB
                color = mcolors.to_rgb(patch_colors[patch_type])
                grid_data[x, y] = color
    
    # Create the figure and plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid_data, origin='lower')
    
    # Add a color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=patch_colors[2], label='HH'),
        Patch(facecolor=patch_colors[1], label='HL'),
        Patch(facecolor=patch_colors[0], label='LL')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set grid lines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.5, alpha=0.3)
    
    # Customize axes
    ax.set_xticks(np.arange(-0.5, model.grid.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, model.grid.height, 1), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add title and labels
    ax.set_title(f"Food Distribution Grid ({model.learning_model} model)")
    
    # Display the agent positions as black dots
    agent_counts = np.zeros((model.grid.height, model.grid.width))
    for agent in model.agents:
        x, y = agent.pos
        agent_counts[y, x] += 1
    
    # Plot agents where they exist
    for y in range(model.grid.height):
        for x in range(model.grid.width):
            if agent_counts[y, x] > 0:
                circle = plt.Circle((x, y), 0.3, color='black', alpha=0.7)
                ax.add_patch(circle)
                # Add count text if more than one agent
                if agent_counts[y, x] > 1:
                    ax.text(x, y, int(agent_counts[y, x]), ha='center', va='center', color='white')
    
    plt.tight_layout()
    return fig, ax