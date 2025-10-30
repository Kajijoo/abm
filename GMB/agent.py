from mesa import Agent
from enum import Enum

class Eat(Enum):
    UNHEALTHY = 0
    HEALTHY = 1

class Exercise(Enum):
    IDLE = 0
    EXERCISE = 1

class Citizen(Agent):
    def __init__(
            self, 
            model,
            initial_exercise,
            initial_eat            
        ):
        super().__init__(model)

        self.exercise = initial_exercise
        self.eat = initial_eat
        self.threshold = 0.5

    def update_exercise(self):
        neighbor_nodes = self.model.grid.get_neighborhood(self.pos, include_center = False)

        exercising_neighbors = [
            agent for agent in self.model.grid.get_cell_list_contents(neighbor_nodes)
            if agent.exercise is Exercise.EXERCISE
        ]

        if len(neighbor_nodes) > 0 and len(exercising_neighbors) / len(neighbor_nodes) > self.threshold:
            self.exercise = Exercise.EXERCISE
        else:
            self.exercise = Exercise.IDLE

    def update_eat(self):
        neighbor_nodes = self.model.grid.get_neighborhood(self.pos, include_center = False)

        eating_neighbors = [
            agent for agent in self.model.grid.get_cell_list_contents(neighbor_nodes)
            if agent.eat is Eat.HEALTHY
        ]

        if len(neighbor_nodes) > 0 and len(eating_neighbors) / len(neighbor_nodes) > self.threshold:
            self.eat = Eat.HEALTHY
        else:
            self.eat = Eat.UNHEALTHY

    def step(self):
        self.update_exercise()
        self.update_eat()