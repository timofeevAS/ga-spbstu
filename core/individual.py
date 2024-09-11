from abc import ABC, abstractmethod
import random


class Individual(ABC):
    def __init__(self, genome: int):
        self.genome = genome

    @abstractmethod
    def mutate(self) -> None:
        pass

    @abstractmethod
    def crossover(self, other: "Individual") -> "Individual":
        """
        :param other: - other individual (second parent)
        :return: - return new individual, with current rules of crossover.
        """
        pass