import random
from collections import deque
from typing import Optional, List

import numpy as np

from core.ga import GeneticAlgorithm
from core.individual import Individual
from core.population import Population
from genetic_prog.basics import TERMINAL_SET, FUNCTION_SET
from genetic_prog.feaso import FEASO_RANGE_X1, FEASO_RANGE_X2, fEaso, plot_f_x1_x2_pgtree
from genetic_prog.treenode import TerminalNode, OperatorNode
from pgtree import PGTree

class PGIndividual:
    def __init__(self, genome: PGTree):
        self.genome = genome

    def mutate(self) -> None:
        """Apply mutation to the individual's genome."""
        # Select a random node in the genome
        random_node = self.genome.get_random_node()

        # If the random node is a TerminalNode, change its value
        if isinstance(random_node, TerminalNode):
            # Choose a new terminal value from the terminal set, ensuring it's different
            new_value = random.choice(TERMINAL_SET)
            random_node.value = new_value

        # If the random node is an OperatorNode, change its operator
        elif isinstance(random_node, OperatorNode):
            # Choose a new operator from the function set, ensuring it's different
            new_operator = random.choice(list(FUNCTION_SET.values()))
            random_node.operator = new_operator

    def crossover(self, other: "PGIndividual") -> "PGIndividual":
        """Perform subtree crossover with another individual."""
        # Create copies of both genomes to avoid modifying the original individuals
        new_genome1 = self.genome.copy()
        new_genome2 = other.genome.copy()

        # Randomly select crossover points in both genomes
        crossover_point1 = new_genome1.get_random_node()
        crossover_point2 = new_genome2.get_random_node()

        # Limit the number of attempts to find compatible nodes
        max_attempts = 10
        attempts = 0

        # Attempt to find compatible crossover points
        while not isinstance(crossover_point1, type(crossover_point2)) and attempts < max_attempts:
            crossover_point1 = new_genome1.get_random_node()
            crossover_point2 = new_genome2.get_random_node()
            attempts += 1

        # If compatible points were found, perform the subtree swap
        if isinstance(crossover_point1, type(crossover_point2)):
            new_genome1.swap_subtrees(crossover_point1, crossover_point2)
        else:
            print("Crossover failed: compatible points not found.")

        # Return the new individual as the offspring
        return PGIndividual(new_genome1)


class PGPopulationFeaso(Population):
    def __init__(self, population_size: int):
        super().__init__(population_size)
        self.etalon = fEaso()
        self.individuals: List[PGIndividual] = []

        self.init_population()


    def init_population(self) -> None:
        for i in range(self.population_size):
            self.individuals.append(PGIndividual(PGTree()))

    def sort_by_fitness(self, reverse:bool=False):
        self.individuals.sort(key=lambda x: self.fitness_function(x), reverse=reverse)

    def fitness_function(self, individual: PGIndividual) -> float:
        """Compare with feaoso function"""
        x1_values = np.linspace(*FEASO_RANGE_X1, num=10)
        x2_values = np.linspace(*FEASO_RANGE_X2, num=10)
        # Initialize the sum of abs differences
        total_error = 0.0
        count = 0

        # Iterate over the grid and compute abs differences
        for x1 in x1_values:
            for x2 in x2_values:
                variables = {'x1': x1, 'x2': x2}
                individual_val = individual.genome.evaluate(variables)
                etalon_val = self.etalon.evaluate(variables)

                # Calculate abs difference
                squared_difference = abs((individual_val - etalon_val))
                total_error += squared_difference
                count += 1

        # Compute the mean squared error
        mae = total_error / count
        return mae

class PGGeneticAlgorithmFeaso(GeneticAlgorithm):
    def __init__(self, population_size: int, crossover_p: float, mutation_p: float, elite_count: int, debug_info: bool = False):
        super().__init__(population_size, crossover_p, mutation_p)

        self.elite_count = elite_count
        self.population = PGPopulationFeaso(population_size)
        self.elite_individuals: List[PGIndividual] = []
        self.debug_mode = debug_info

    def debug(self, info: str):
        if self.debug_mode is False:
            return

        print(f'[DEBUG]: {info}')

    def save_elite(self) -> None:
        self.population.sort_by_fitness()
        self.elite_individuals = self.population[:self.elite_count]

    def clear_invalid(self):
        for ind in self.population.individuals:
            if not ind.genome.is_correct():
                self.population.individuals.remove(ind)

    def round(self) -> None:
        # 0. Remove incorrects.
        self.clear_invalid()

        # 1. Selection:
        self.debug('Start selection')
        self.selection(10, 5)

        self.debug('Start reproduction')
        # 2. Reproduction.
        self.reproduction()

        # 3. Mutation.
        self.mutation() # mutation inside reproduction

        self.debug('Start reduction')
        # 4. Reduction.
        self.reduction()

    def selection(self, group_size: int = 2, top_count: int = 1) -> None:
        # Tournament selection;
        tmp_individuals = self.population.individuals.copy()
        random.shuffle(tmp_individuals)

        # Get groups for tournament selection.
        groups = []
        tmp_individuals = deque(tmp_individuals)
        while tmp_individuals:
            group = []
            for i in range(group_size):
                if not tmp_individuals:
                    break
                individ = tmp_individuals.pop()
                group.append(individ)

            groups.append(group)

        tmp_individuals = []
        for g in groups:
            g.sort(key=lambda x: self.population.fitness_function(x))
            tmp_individuals += g[:top_count]

        self.population.individuals = tmp_individuals


    def reproduction(self) -> None:
        p = self.population
        childs = []

        while len(childs) < self.population_size:
            if random.random() <= self.crossover_p:
                parent1, parent2 = random.choices(self.population.individuals, k=2)

                c1 = parent1.crossover(parent2)

                # Mutation for child
                if random.random() <= self.mutation_p:
                    c1.mutate()

                childs.append(c1)
        p.individuals.extend(childs)

    def mutation(self) -> None:
        pass

    def reduction(self) -> None:
        self.population.individuals.extend(self.elite_individuals)
        self.population.sort_by_fitness()
        self.population.individuals = self.population.individuals[:self.population_size]

    def run(self) -> None:
        raise NotImplemented("Still not implemented")

    def run_for(self, iteration: int):
        for i in range(iteration):
            self.round()
            print(f'Round {i} finished.')

    def get_best(self) -> PGIndividual:
        self.population.sort_by_fitness()
        return self.population.individuals[0]

if __name__ == '__main__':
    ga = PGGeneticAlgorithmFeaso(50, 0.5, 0.01, 5)
    ga.run_for(100)

    best = ga.get_best()
    print(f'Mae: {ga.population.fitness_function(best)}')
    print(f'f: {best.genome.root}')
    plot_f_x1_x2_pgtree(best.genome)
