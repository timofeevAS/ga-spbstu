import random
from typing import List
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.function import OneParamFunction
from core.ga import GeneticAlgorithm
from core.individual import Individual
from core.population import Population
from utils.number_present import *

DEFAULT_PRECISION = 3
DEFAULT_CROSSOVER_PROBABILITY = 0.5  # Probabilitiy of crossover two individs.
DEFAULT_MUTATION_PROBABILITY = 0.01  # Probability of mutation via crossover.


class NumberInRange(Individual):
    def __init__(self, value: int, a: float, b: float, precision: int = DEFAULT_PRECISION):
        super().__init__(value)
        self.a = a
        self.b = b
        self.precision = precision

    def mutate(self) -> None:
        binary_str = from_int_to_binary_string(self.genome)
        mutate_idx = random.randint(0, len(binary_str) - 1)

        self.genome ^= (1 << mutate_idx)

    def crossover(self, other: "NumberInRange") -> "NumberInRange":
        """
        Single-point crossover implementation.
        :param other: - other individual
        :return: - new individul
        """
        binary_str1 = from_int_to_binary_string(self.genome)
        binary_str2 = from_int_to_binary_string(other.genome)

        binary_str1, binary_str2 = synchronize_binary_strings(binary_str2, binary_str1)
        genome_len = len(binary_str1)
        crossover_idx = random.randint(0, genome_len - 1)

        child_genome = binary_str1[:crossover_idx] + binary_str2[crossover_idx:]
        return NumberInRange(int(child_genome, 2), self.a, self.b, self.precision)

    def get_real_value(self) -> float:
        binary_str = from_int_to_binary_string(self.genome)
        real_value = binary_arithmetic_number(binary_str, self.a, self.b, self.precision)
        return real_value

    def __repr__(self):
        return f"[i] {self.genome} -> {from_int_to_binary_string(self.genome)}"


class NumberPopulation(Population):

    def __init__(self, population_size: int, a: float, b: float,
                 fitness: OneParamFunction,
                 precision: int = DEFAULT_PRECISION):
        super().__init__(population_size)
        self.individuals: List[NumberInRange] = []
        self.a = a
        self.b = b
        self.fitness = fitness
        self.precision = precision
        self.init_population()

    def init_population(self) -> None:
        max_value = 2 ** (auto_precision(self.a, self.b, self.precision))  # Max value is 2^N as 111..111, series of 1.
        for i in range(self.population_size):
            random_genome = random.randint(0, max_value)
            individual = NumberInRange(random_genome, self.a, self.b, self.precision)
            self.individuals.append(individual)

    def fitness_function(self, individual: NumberInRange) -> float:
        value = self.fitness.evaluate(individual.get_real_value())
        return value

    def __str__(self) -> str:
        res = ""
        for i in range(self.population_size):
            individ = self.individuals[i]
            res += (f'[{i + 1}]: {individ.get_real_value()} -> {self.fitness_function(individ)}\n')
        return res


class FunctionMinMax(GeneticAlgorithm):
    def __init__(self,
                 population_size: int,
                 left_border: float,
                 right_border: float,
                 function: OneParamFunction,
                 precision: int = DEFAULT_PRECISION,
                 crossover_p: float = DEFAULT_CROSSOVER_PROBABILITY,
                 mutation_p: float = DEFAULT_MUTATION_PROBABILITY):
        super().__init__(population_size, crossover_p, mutation_p)

        self.population: NumberPopulation = NumberPopulation(population_size,
                                                             left_border,
                                                             right_border,
                                                             function,
                                                             precision)
        self.function = function

    def round(self) -> None:
        # Any round of genetic algorithm starts with:
        # 1. Selection.
        # 2. Reproduction.
        # 3. Mutation.
        # 4. Reduction (to initial size).

        # 1. Selection:
        self.selection()

        # 2. Reproduction.
        self.reproduction()

        # 3. Mutation.
        self.mutation()

        # 4. Reduction.
        self.reduction()

    def selection(self) -> None:
        p = self.population
        p.individuals.sort(key=lambda x: p.fitness_function(x))

        # Here implementation roulette method.
        fitness_values = list(map(p.fitness_function, p.individuals.copy()))
        sum_f: float = sum(fitness_values)
        probabilities = list(map(lambda x: x / sum_f, fitness_values))

        # Get transition population for crossovering.
        transition_population: List[NumberInRange] = random.choices(p.individuals,
                                                                    weights=probabilities,
                                                                    k=p.population_size)
        p.individuals = transition_population

    def reproduction(self) -> None:
        p = self.population
        for i in range(p.population_size):
            if random.random() <= self.crossover_p:
                # Get two random parents from population.
                parent1: NumberInRange = p.individuals[random.randint(0, p.population_size - 1)]
                parent2: NumberInRange = p.individuals[random.randint(0, p.population_size - 1)]

                # New individual into population.
                child = parent1.crossover(parent2)
                p.individuals.append(child)

    def mutation(self) -> None:
        p = self.population
        for i in range(p.population_size):
            if random.random() <= self.mutation_p:
                p.individuals[i].mutate()

    def reduction(self) -> None:
        p = self.population
        p.individuals.sort(key=lambda x: p.fitness_function(x), reverse=True)
        p.individuals = p.individuals[:p.population_size]

    def run(self, iteration_count: int) -> None:
        print('init:')
        print(self.population)
        for i in range(iteration_count):
            print(f'Round: {i}:\n{self.population}')
            self.round()


if __name__ == '__main__':
    # TODO: remove. It s just testing function.
    f = OneParamFunction("((exp(x)-exp(-x))*cos(x))/(exp(x)+exp(-x))", "x")
    mx = FunctionMinMax(100, -5, 5, f)
    mx.run(250)
