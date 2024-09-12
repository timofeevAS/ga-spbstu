from abc import ABC, abstractmethod

from core.population import Population


class GeneticAlgorithm(ABC):
    def __init__(self, population_size: int, crossover_p: float, mutation_p: float):
        self.population_size = population_size
        self.crossover_p = crossover_p
        self.mutation_p = mutation_p

    @abstractmethod
    def round(self) -> None:
        """
        Method to 'play' round of genetic algorithm.
        :return:
        """
        pass

    @abstractmethod
    def selection(self) -> None:
        """
        Method to selection step of genetic algorithm.
        :return:
        """

    @abstractmethod
    def reproduction(self) -> None:
        """
        Method to reproduction step of genetic algorithm.
        :return:
        """
        pass

    @abstractmethod
    def mutation(self) -> None:
        """
        Method to mutation step of genetic algorithm.
        :return:
        """
        pass

    @abstractmethod
    def reduction(self) -> None:
        """
        Method to reduction step of genetic algorithm.
        :return:
        """
        pass

    @abstractmethod
    def run(self) -> None:
        """
        Method to run genetic algorithm.
        :return:
        """