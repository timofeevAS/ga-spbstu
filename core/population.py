from abc import ABC, abstractmethod
from typing import List
import random

from core.individual import Individual


class Population(ABC):
    def __init__(self, population_size: int):
        self.population_size = population_size
        self.individuals: List[Individual] = []

    @abstractmethod
    def init_population(self) -> None:
        pass

    @abstractmethod
    def fitness_function(self, individual: Individual) -> float:
        pass