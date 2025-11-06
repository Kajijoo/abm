from mesa import Agent
import random

class LearningAgent(Agent):
    def __init__(self, model, row, learning_model, epsilon, value_low, value_high, p_low, p_high):
        super().__init__(model)
        self.row = row
        self.learning_model = learning_model  # 'RW' or 'TD' or 'RWE"
        self.learning_rate = 0.8 # Rate of learning
        self.extinction_rate = 1.0 # Standard for RW extinction = 1, typically <1
        self.delta = 0.0 # Standard for delta = 0. Standard logstic for delta = 1, S-curve 0 < delta < 1
        self.beta = 1.0 # Responsivity to food in TD learning
        self.value_low = value_low  # Initial value outcome
        self.value_high = value_high # Initial value outcome
        self.food_consumed = None  # Food consumed status ('L' or 'H')
        self.foods_consumed = {"H": 0, "L": 0}   # Foods consumed
        self.p_low = p_low  # True reward value of food type L
        self.p_high = p_high  # True reward value of food type H
        self.epsilon = epsilon

    def move(self): # Move agent to the right in grid space
        x, y = self.pos
        new_x = (x + 1) % self.model.grid.width
        self.model.grid.move_agent(self, (new_x, y))

    def eat(self):
        ptype = self.model.grid.properties["patch_type"].data[self.pos]

        if ptype == 0:  # LL
            self.food_consumed = "L"
        elif ptype == 2:  # HH
            self.food_consumed = "H"
        elif self.value_low == self.value_high:
            self.food_consumed = random.choice(["H", "L"])
        else:
            if self.value_low < self.value_high:
                self.food_consumed = "L" 
            else:
                self.food_consumed = "H"
            # add exploration noise
            if random.random() < self.epsilon:
                self.food_consumed = "L" if self.food_consumed == "H" else "H"

        self.foods_consumed[self.food_consumed] += 1  # update counts

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