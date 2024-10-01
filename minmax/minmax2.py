import math
import random
import time
from typing import List, Tuple, Union
import sys
import os

from examples.branins_rcos_function import BraninsRcosFunction

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.function import  TwoParamFunction
from core.ga import GeneticAlgorithm
from core.individual import  RGAIndividual
from core.population import  RGAPopulation

DEFAULT_CROSSOVER_PROBABILITY = 0.9  # Probability of crossover two individs.
DEFAULT_MUTATION_PROBABILITY = 0.1  # Probability of mutation via crossover.
DEFAULT_ELITE_COUNT = 10 # Count of `elite` individuals left in transition population.


class RGANumber(RGAIndividual):

    def __init__(self, arg1: Union[int, List], arg2: List[Tuple[float, float]] = None):
        if isinstance(arg1, list):
            self.genome = arg1
            self.dimension = len(self.genome)
            self.ranges = None

        elif isinstance(arg1, int) and arg2 is not None:
            super().__init__(arg1, arg2)
        else:
            raise ValueError("Incorrect args")

    def mutate(self) -> None:
        pass

    def uneven_mutate(self, current_iteration: int, total_iteration: int, B: float = 1, CL: float = -1, CR: float = 1) -> None:
        """
            Uneven mutation implementation
                    { * c + d(t, cr - c)         a <= 0.5
            c_m =>  {
                    { * c + d(t, ck - cl)        a > 0.5
            a = random.random()
            d(t, y) = y * (1 - pow(r, (1 - (t / T)) ** b)
            r = random.random()
            b = const (can be 1)
            T = total count of iteration
            t = current iteration
            :return:
        """

        def delta(y: float, t: int = current_iteration, T: int = total_iteration, b: float = B) -> float:
            r = random.random()
            return y * (1 - pow(r, (1 - (t / T)) ** b))

        gen_idx = random.randint(0, self.dimension - 1)

        gen = self.genome[gen_idx]
        mutated_gen: float

        a = random.random()
        if a <= 0.5:
            mutated_gen = gen + delta(CR-gen)
        else:
            mutated_gen = gen - (delta(gen-CL))

        self.genome[gen_idx] = mutated_gen

    def crossover(self, other: "RGANumber") -> "RGANumber":
        # TODO: make difference way to crossover:
        # - SBX
        # - Min-Max
        # - Wright’s heuristic crossover)
        # etc...
        return self.flat_crossover(other)

    def flat_crossover(self, other: "RGANumber") -> "RGANumber":
        child_genome: List = []
        for i in range(self.dimension):
            random_ci = random.uniform(self.genome[i], other.genome[i])
            child_genome.append(random_ci)

        return RGANumber(child_genome)

    def sbx_crossover(self, other: "RGANumber", n: int = 2) -> Tuple["RGANumber", "RGANumber"]:
        def sbx_beta(nu: int) -> float:
            b: float
            power_val = (1 / (nu + 1))

            u = random.random()
            if u <= 0.5:
                b = (2 * u) ** power_val
            else:
                b = (1 / (2 * (1 - u))) ** power_val
            return b


        beta = sbx_beta(n)

        # Crossover child 1 and 2.
        child1_genome: List = []
        child2_genome: List = []
        for i in range(self.dimension):
            # TODO: is need to make floor in this case: 0.5 * floor(1-beta)*p1_i...
            p1_gen = self.genome[i]
            p2_gen = other.genome[i]

            child1_genome.append(0.5 * ((1 - beta) * p1_gen + (1 + beta) * p2_gen))
            child2_genome.append(0.5 * ((1 + beta) * p1_gen + (1 - beta) * p2_gen))

        return RGANumber(child1_genome), RGANumber(child2_genome)

    def get_real_value(self) -> List:
        return self.genome

    def __repr__(self):
        return f"[i] {self.genome}"


class RGATwoParamFuncPopulation(RGAPopulation):

    def __init__(self, population_size: int, fitness_func: TwoParamFunction,
                 ranges: List[Tuple[float, float]]):
        super().__init__(population_size)
        self.individuals: List[RGANumber] = []
        self.fitness = fitness_func
        self.ranges = ranges
        self.init_population()


    def init_population(self) -> None:
        for i in range(self.population_size):
            individ: RGANumber = RGANumber(2, self.ranges)
            self.individuals.append(individ)

    def fitness_function(self, individual: "RGANumber") -> float:
        return self.fitness.evaluate(*individual.get_real_value())

class FunctionMinMax2(GeneticAlgorithm):
    def __init__(self,
                 population_size: int,
                 function: TwoParamFunction,
                 ranges: List[Tuple[float, float]],
                 crossover_p: float = DEFAULT_CROSSOVER_PROBABILITY,
                 mutation_p: float = DEFAULT_MUTATION_PROBABILITY,
                 elite_c: int = DEFAULT_ELITE_COUNT):
        super().__init__(population_size, crossover_p, mutation_p)

        self.population = RGATwoParamFuncPopulation(self.population_size, function, ranges)

        self.function = function
        self.ranges = ranges
        self.elite_c = elite_c
        self.current_iter = 0

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
        """
        Create selection with roulette and "elitarity" principe.
        :return:
        """
        p = self.population
        p.individuals.sort(key=lambda x: p.fitness_function(x))

        # Add elite individuals.
        transition_population: List[RGANumber] = []
        for i in range(self.elite_c):
            transition_population.append(p.individuals[i])


        # Here implementation roulette method.
        fitness_values = list(map(p.fitness_function, p.individuals.copy()))
        sum_f: float = sum(fitness_values)
        probabilities = list(map(lambda x: x / sum_f, fitness_values))

        # Get transition population for crossovering.
        transition_population.extend(random.choices(p.individuals, weights=probabilities, k=p.population_size))
        p.individuals = transition_population

        p.individuals.sort(key=lambda x: p.fitness_function(x))

    def reproduction(self) -> None:
        p = self.population
        for i in range(p.population_size):
            if random.random() <= self.crossover_p:
                # Get two random parents from population.
                parent1: RGANumber
                parent2: RGANumber

                parent1, parent2 = random.choices(p.individuals, k=2)
                # New individual into population.
                child1, child2 = parent1.sbx_crossover(parent2)

                # Append children into population
                p.individuals.extend((child1, child2))


    def mutation(self) -> None:
        p = self.population

        individ: RGANumber
        for individ in p.individuals:
            if random.random() <= self.mutation_p:
                individ.uneven_mutate(self.current_iter, self.total_iter)


    def reduction(self) -> None:
        p = self.population
        p.individuals.sort(key=lambda x: p.fitness_function(x))
        p.individuals = p.individuals[:p.population_size]


    def run(self, iteration_count: int) -> None:
        print('init:')
        print(self.population)
        self.total_iter = iteration_count
        for i in range(iteration_count):
            print(f'Round: {i}:\n{self.population}')
            self.round()
            self.current_iter += 1

    def get_individs(self) -> List[RGANumber]:
        return self.population.individuals

    def get_best(self) -> RGANumber:
        p = self.population
        p.individuals.sort(key=lambda x: p.fitness_function(x))
        return self.population.individuals[0]

if __name__ == '__main__':
    ga1 = FunctionMinMax2(50, BraninsRcosFunction(), [(2.5, 10), (7.5, 15)])
    ga1.run(40)

    ga2 = FunctionMinMax2(50, BraninsRcosFunction(), [(-5, 2.5), (0, 7.5)])
    ga2.run(40)

    ga3 = FunctionMinMax2(50, BraninsRcosFunction(), [(-5, 2.5), (7.5, 15)])
    ga3.run(40)

    ga4 = FunctionMinMax2(50, BraninsRcosFunction(), [(2.5, 10), (7.5, 15)])
    ga4.run(40)

    # Collect result:
    dots = []
    for rgan in ga1.get_individs():
        coords = tuple(rgan.get_real_value())
        dots.append(coords)

    for rgan in ga2.get_individs():
        coords = tuple(rgan.get_real_value())
        dots.append(coords)

    for rgan in ga3.get_individs():
        coords = tuple(rgan.get_real_value())
        dots.append(coords)

    for rgan in ga4.get_individs():
        coords = tuple(rgan.get_real_value())
        dots.append(coords)


    print(ga1.get_best())
    print(ga2.get_best())
    print(ga3.get_best())
    print(ga4.get_best())

    print(ga1.function.evaluate(*ga1.get_best().get_real_value()))
    print(ga2.function.evaluate(*ga2.get_best().get_real_value()))
    print(ga3.function.evaluate(*ga3.get_best().get_real_value()))
    print(ga4.function.evaluate(*ga4.get_best().get_real_value()))

    BraninsRcosFunction().print_plot_with_points(dots)