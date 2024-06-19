from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class LearningAgent(Agent):
    def __init__(self, unique_id, model, row, learning_model):
        super().__init__(unique_id, model)
        self.row = row
        self.learning_model = learning_model  # 'RW' or 'TD'
        self.learning_rate = 0.1 # Rate of learning
        self.extinction_rate = 1.0 # Standard for RW extinction = 1, typically <1
        self.delta = 0.0 # Standard for delta = 0. Standard logstic for delta = 1, S-curve 0 < delta < 1
        self.beta = 1.0 # Responsivity to food in TD learning
        self.value_low = 0.0  # Initial value outcome
        self.value_high = 0.0 # Initial value outcome
        self.ptype = None  # The current patch color where the agent is located
        self.food_consumed = None  # Food consumed status ('L' or 'H')
        self.p_low = 0.2  # True reward value of food type L
        self.p_high = 0.8  # True reward value of food type H
        #self.bmi = bmi

    def move(self): # Move agent to the right in grid space
        x, y = self.pos
        new_x = (x + 1) % self.model.grid.width
        self.model.grid.move_agent(self, (new_x, y))

    def eat(self): #Eat procedure for TDW copied from the Hammond et al. 2012 NetLogo model
        #Get current patch type
        self.ptype = self.model.grid.get_cell_list_contents([self.pos])[0].type

        # Determine food consumed based on patch type
        if self.ptype in ["HH", "LL"]:
            self.food_consumed = self.ptype[0]
        elif self.value_low == self.value_high:
            self.food_consumed = random.choice(["H", "L"])
        else:
            if self.value_low > self.value_high:
                self.food_consumed = "L"
            else:
                self.food_consumed = "H"
            if random.random() < 0.05:
                self.food_consumed = "L" if self.food_consumed == "H" else "H"

#This is the updated RW so it is similar to TD, but we need to think about when and how extinction kicks in.
#Right now extinction for opposite food (H or L) kicks in when one is consumed, because there is no situation where nothing is consumed.
    def update_affect_rw(self):
        if self.food_consumed == "L":
            self.value_low = self.value_low + (self.learning_rate * (self.value_low ** self.delta) * (self.p_low - self.value_low))
            self.value_high = self.value_high + (self.learning_rate * (self.value_high ** self.delta) * self.extinction_rate * (0 - self.value_high))
        else:
            self.value_high = self.value_high + (self.learning_rate * (self.value_high ** self.delta) * (self.p_high - self.value_high))
            self.value_low = self.value_low + (self.learning_rate * (self.value_low ** self.delta) * self.extinction_rate * (0 - self.value_low))  

    def update_reward_td(self):
        if self.food_consumed == 'L':
            self.value_low = (self.value_low + self.learning_rate * (self.beta * self.p_low - self.value_low))
        else:
            self.value_high = (self.value_high + self.learning_rate * (self.beta * self.p_high - self.value_high))


#Get B responsivity based on BMI. @Yara, can you show me how to rewrite this into one line? 
    def get_beta(self):
        if self.bmi < 18.5:
            return 1.25
        elif 18.5 <= self.bmi <= 24.9:
            return 1
        elif 25 <= self.bmi <= 29.9:
            return 0.75
        else:
            return 0.5

#THIS WILL BE FOR 2ND PAPER. INTRODUCING BETA IN RESCORLA WAGNER MODEL FOR SUBJECTIVE REWARD EXPERIENCE BASED ON BMI.

    def update_affect_rw_b(self):
        self.get_beta()
        if self.food_consumed == "L":
            self.value_low = self.value_low + (self.learning_rate * (self.value_low ** self.delta) * (self.beta * self.p_low - self.value_low))
            self.value_high = self.value_high + (self.learning_rate * (self.value_high ** self.delta) * self.extinction_rate * (0 - self.value_high))
        else:
            self.value_high = self.value_high + (self.learning_rate * (self.value_high ** self.delta) * (self.beta * self.p_high - self.value_high))
            self.value_low = self.value_low + (self.learning_rate * (self.value_low ** self.delta) * self.extinction_rate * (0 - self.value_low))


    def step(self): 
        self.move()
        
        #print(f"I ate {str(self.food_consumed)}")
        #print(f"My TD reward learning is: {str(self.value_high)}")
        #print(f"My RW reward learning is: {str(self.affect)}")

        #Update learning
        if self.learning_model == 'RW':
            self.update_affect_rw()
            self.eat()
        elif self.learning_model == 'TD':
            self.update_reward_td()
            self.eat()

class Patch(Agent):
    def __init__(self, unique_id, model, patch_type):
        super().__init__(unique_id, model)
        self.type = patch_type

    def get_color(self):
        if self.type == "HH":
            return "red"
        elif self.type == "LL":
            return "blue"
        elif self.type == "HL":
            return "purple"
        return "white"

class LearningModel(Model):
    def __init__(self, N, width, height, learning_model='RW', distribute_patches = 'random', seed = None):
        super().__init__()
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = SimultaneousActivation(self)
        self.learning_model = learning_model
        
        if seed is not None:
            random.seed(seed)
            self.random.seed(seed)

        #Create agents 
        for i in range(self.num_agents):
            agent = LearningAgent(i, self, row=i, learning_model=learning_model)
            self.grid.place_agent(agent, (0, i))
            self.schedule.add(agent)

        #Add patches with types based on different distributions
        if distribute_patches == 'random':
            self.distribute_randomly()
        elif distribute_patches == 'gradient_h':
            self.distribute_gradient_h()
        elif distribute_patches == 'gradient_l':
            self.distribute_gradient_l()

        self.datacollector = DataCollector(
            agent_reporters={"Value_Low": "value_low", "Value_High": "value_high"}
        )

    def distribute_randomly(self):
        patch_types = ["HH", "LL", "HL"]
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                patch_type = random.choice(patch_types)
                patch = Patch(f'patch_{x}_{y}', self, patch_type)
                self.grid.place_agent(patch, (x,y))

    def distribute_gradient_h(self):
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                prob_hh = x / self.grid.width
                if random.random() < prob_hh:
                    patch_type = "HL"
                else:
                    patch_type = "HH"
                patch = Patch(f'patch_{x}_{y}', self, patch_type)
                self.grid.place_agent(patch, (x, y))

    def distribute_gradient_l(self):
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                prob_ll = x / self.grid.width
                if random.random() < prob_ll:
                    patch_type = "HL"
                else:
                    patch_type = "LL"
                patch = Patch(f'patch_{x}_{y}', self, patch_type)
                self.grid.place_agent(patch, (x, y))

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
        cmap = mcolors.ListedColormap(['red', 'blue', 'purple', 'white'])
        bounds = [0, 1, 2, 3, 4]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        matrix = [[bounds.index(cmap.colors.index(color)) for color in row] for row in grid_matrix]
        ax.imshow(matrix, cmap = cmap, norm = norm)

        plt.show()   