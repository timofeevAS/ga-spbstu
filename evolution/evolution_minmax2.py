import concurrent.futures
import json
import random
import time
from typing import List, Tuple, Union
import sys
import os

import numpy as np
from numpy.ma.extras import average

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from examples.functions.branins_rcos_function import BraninsRcosFunction
from utils.number_present import in_range

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.function import  TwoParamFunction
from core.ga import GeneticAlgorithm
from core.individual import  RGAIndividual
from core.population import  RGAPopulation

DEFAULT_CROSSOVER_PROBABILITY = 0.9  # Probability of crossover two individs.
DEFAULT_MUTATION_PROBABILITY = 0.1  # Probability of mutation via crossover.
DEFAULT_ELITE_COUNT = 5 # Count of `elite` individuals left in transition population.


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
        raise NotImplemented("Not implemented")

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

class EvolutionMinMax2(GeneticAlgorithm):
    def __init__(self,
                 population_size: int,
                 function: TwoParamFunction,
                 ranges: List[Tuple[float, float]],
                 lambda_value: int,
                 nu_value: int,
                 min_sigma=0.1,
                 max_sigma=1):
        super().__init__(population_size, 0, 0)

        self.start_time = 0
        self.population = RGATwoParamFuncPopulation(self.population_size, function, ranges)

        self.function = function
        self.ranges = ranges
        self.t = 0 # Iteratioon count
        self.lambda_value = lambda_value # Number of childs which NU_VALUE create;
        self.nu_value = nu_value # Number of parents which create LAMBDA_value childs;

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.sigma = self.init_sigma_vector()

        self.results = {'steps':[], 'nu':self.nu_value,
                        'lambda':self.lambda_value,
                        'min_sigma':self.min_sigma,
                        'max_sigma':self.max_sigma}

        if lambda_value/nu_value < 7:
            raise ValueError(f"Lambda_value/nu_value should be greater than 7,\n but suggest: {lambda_value}/{nu_value}={lambda_value/nu_value}")

        self.total_mutations_count = 0
        self.success_mutations_count = 0


    def save_info(self):
        points = []
        for ind in self.population.individuals:
            points.append(ind.genome)
        info = {'step':self.t,
                'time':time.time() - self.start_time,
                'best': self.population.fitness_function(self.get_best()),
                'points': points}
        self.results['steps'].append(info)

    def sigma_balanced(self):
        def fi_k():
            return self.success_mutations_count / self.total_mutations_count

        c_d = 0.82
        c_i = 1/0.82 # 1.22

        for sigma_component in self.sigma:
            w = 1
            fk = fi_k()
            if fk < 0.2:
                w = c_d
            elif fk > 0.2:
                w = c_i
            elif fk - 0.2 < 0.01:
                w = 1
            sigma_component *= w


    def round(self) -> None:
        # Any round of evolution algorithm:
        # Reproduction: -> generate childs and mutate them

        # 2. Reproduction.
        self.reproduction()

        # 3. Reduction.
        self.reduction()

        self.sigma_balanced()

        self.t+=1
        self.save_info()

    def selection(self) -> None:
        """
        Create selection with roulette and "elitarity" principe.
        :return:
        """
        raise NotImplemented("Evolution strategy not implement selection.")

    def init_sigma_vector(self):
        return np.random.uniform(self.min_sigma, self.max_sigma, len(self.ranges))

    def N_random(self, i):
        return np.random.normal(0, self.sigma[i])

    def reproduction(self) -> None:
        p = self.population
        childs: List[RGANumber] = []

        for i in range(self.lambda_value):
            # Get NU random parents
            parents: List[RGANumber]

            parents = random.choices(p.individuals, k=self.nu_value)
            child_genome: List[float] = []
            # Crossovering
            for j in range(len(self.ranges)):
                # Take random component of parents
                randp: RGANumber = random.choice(parents)
                value = randp.genome[j] + self.N_random(j)
                child_genome.append(value)

            # Mutate genome
            for j in range(len(self.ranges)):
                child_genome[j] += self.N_random(j)

            # New child
            child: RGANumber = RGANumber(child_genome)
            self.total_mutations_count +=1

            # Compare with parents
            for parent in parents:
                if self.population.fitness_function(child) < self.population.fitness_function(parent):
                    self.success_mutations_count += 1

            childs.append(child)

        self.population.individuals.extend(childs)

    def mutation(self) -> None:
        raise NotImplemented("Not implemented there")


    def reduction(self) -> None:
        p = self.population
        p.individuals.sort(key=lambda x: p.fitness_function(x))
        p.individuals = p.individuals[:p.population_size]


    def run(self, iteration_count: int, filename = 'evolution.json') -> None:
        print(f'Started {self.__hash__()}')
        self.start_time = time.time()
        for i in range(iteration_count):
            self.round()
        print(f'Finished {self.__hash__()}')

        with open(filename, "w", encoding="utf-8") as json_file:
            json.dump(self.results, json_file, ensure_ascii=False, indent=4)

    def get_individs(self) -> List[RGANumber]:
        return self.population.individuals

    def get_best(self) -> RGANumber:
        p = self.population
        p.individuals.sort(key=lambda x: p.fitness_function(x))
        return self.population.individuals[0]

def experimentA():
    # 100 round with next params
    POPULATION_SIZE=50
    MIN_SIGMA=0.1
    MAX_SIGMA=1
    LAMBDA = 40
    NU = 5
    for i in range(25):
        ga = EvolutionMinMax2(POPULATION_SIZE, BraninsRcosFunction(), [(-5, 10), (0, 15)], LAMBDA, NU)
        ga.run(100, f'evolution_{i}.json')
        print(f'Finished {i} from {50}')

if __name__ == '__main__':
    experimentA()

    with open('evolution.json', 'r') as file:
        data = json.load(file)

    pnts = data['steps'][-1]['points']
    BraninsRcosFunction().print_plot_with_points(pnts)
