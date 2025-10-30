import mesa
import networkx as nx
from mesa import Model
from agent import Citizen, Eat, Exercise


def number_state_eat(model, eat):
    return sum(1 for a in model.grid.get_all_cell_contents() if a.eat is eat)

def number_state_exercise(model, exercise):
    return sum(1 for a in model.grid.get_all_cell_contents() if a.exercise is exercise)

def number_combined_state(model, eat, exercise):
    return sum(1 for a in model.grid.get_all_cell_contents() if a.eat is eat and a.exercise is exercise)

def number_exercising(model):
    return number_state_exercise(model, Exercise.EXERCISE)

def number_idle(model):
    return number_state_exercise(model, Exercise.IDLE)

def number_healthy(model):
    return number_state_eat(model, Eat.HEALTHY)

def number_unhealthy(model):
    return number_state_eat(model, Eat.UNHEALTHY)

def healthy_exercisers(model):
    return number_combined_state(model, Eat.HEALTHY, Exercise.EXERCISE)

def unhealthy_exercisers(model):
    return number_combined_state(model, Eat.UNHEALTHY, Exercise.EXERCISE)

def healthy_idle(model):
    return number_combined_state(model, Eat.HEALTHY, Exercise.IDLE)

def unhealthy_idle(model):
    return number_combined_state(model, Eat.UNHEALTHY, Exercise.IDLE)
  


class NetworkModel(Model):
    def __init__(self, num_nodes = 100, initial_healthy = 10, seed = None):
        super().__init__(seed = seed)
        self.num_nodes = num_nodes
        self.G = nx.barabasi_albert_graph(num_nodes, m = 1)
        self.grid = mesa.space.NetworkGrid(self.G)

        if initial_healthy > num_nodes:
            self.initial_healthy = num_nodes
        else:
            self.initial_healthy = initial_healthy
    
        self.datacollector = mesa.DataCollector(
            {"Idle": number_idle,
             "Exercising": number_exercising,
             "Healthy eating": number_healthy,
             "Unhealthy eating": number_unhealthy,
             "Healthy exercisers": healthy_exercisers, 
             "Unhealthy exercisers": unhealthy_exercisers,
             "Healthy idle": healthy_idle,
             "Unhealthy idle": unhealthy_idle
             }
        )

        for node in self.G.nodes():
            a = Citizen(
                self,
                Exercise.IDLE,
                Eat.UNHEALTHY,

            )
            self.grid.place_agent(a, node)

        exercising_nodes = self.random.sample(list(self.G), self.initial_healthy)
        for a in self.grid.get_cell_list_contents(exercising_nodes):
            a.exercise = Exercise.EXERCISE
        
        eating_nodes = self.random.sample(list(self.G), self.initial_healthy)
        for a in self.grid.get_cell_list_contents(eating_nodes):
            a.eat = Eat.HEALTHY

        self.running = True
        self.datacollector.collect(self)
        
    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)