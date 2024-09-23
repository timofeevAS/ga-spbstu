from abc import ABC, abstractmethod
import random
from typing import List, Tuple


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

class RGAIndividual(ABC):
    def __init__(self, dimension: int, ranges: List[Tuple[float, float]]):
        self.dimension = dimension
        self.ranges = ranges
        self.genome: List = []

        # Generate random value in genome
        for i in range(dimension):
            ci = random.uniform(ranges[i][0], ranges[i][1])
            self.genome.append(ci)

    @abstractmethod
    def mutate(self) -> None:
        pass

    @abstractmethod
    def crossover(self, other: "RGAIndividual") -> "RGAIndividual":
        """
        :param other: - other individual (second parent)
        :return: - return new individual, with current rules of crossover.
        """
        pass

    def __str__(self):
        gens = ''
        for c in self.genome:
            gens+=f'{c}, '

        gens=gens[:len(gens)-2]
        return f'({gens})'

