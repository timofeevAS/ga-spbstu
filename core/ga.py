from abc import ABC, abstractmethod

from core.population import Population


class GeneticAlgorithm(ABC):
    def __init__(self, population: Population):
        self.population = population
        population.init_population()

    @abstractmethod
    def round(self) -> None:
        """
        Method to 'play' round of genetic algorithm.
        :return:
        """
        pass

    def selection(self) -> None:
        """
        Method to selection step of genetic algorithm.
        :return:
        """