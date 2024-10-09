import random
import unittest
from typing import List

from core.individual import Individual
from core.population import Population
from tsp.tsp1 import OrdinalTourIndividual, TSPPopulationOrdinalAdjacencyMatrix


class TSP1Tests(unittest.TestCase):
    def test_ordinal_individual(self):
        ordinal_tour = [0, 0, 1, 0, 3, 0, 2, 0, 0]
        ordinal = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        individual = OrdinalTourIndividual(ordinal_tour, ordinal)

        expectedTour = [0, 1, 3, 2, 7, 4 ,8, 5, 6]

        self.assertEqual(expectedTour, individual.get_tour())

    def test_ordinal_crossover(self):
        ordinal = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        ordinal_tour = [0, 0, 1, 0, 3, 0, 2, 0, 0]
        individual1 = OrdinalTourIndividual(ordinal_tour, ordinal)

        ordinal_tour = [4, 3, 2, 1, 0, 0, 0, 0, 0]
        individual2 = OrdinalTourIndividual(ordinal_tour, ordinal)

        random.seed(555) # Always get crossover idx = 3.

        childs = individual1.crossover(individual2)
        self.assertEqual(childs[0].genome, [0, 0, 1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(childs[1].genome, [4, 3, 2, 0, 3, 0, 2, 0, 0])

    def test_ordinal_mutate(self):
        ordinal = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        ordinal_tour = [0, 0, 1, 0, 3, 0, 2, 0, 0]
        individual1 = OrdinalTourIndividual(ordinal_tour, ordinal)

        random.seed(555)  # Always get random indexes = 2, 3.

        individual1.mutate() # Mutation. Swap genome[2] and genome[3] 1 <-> 0
        self.assertEqual(individual1.genome, [0, 0, 0, 1, 3, 0, 2, 0, 0])

    def test_population_init(self):
        adjacency_matrix = [
            [0, 10, 15, 20, 10],
            [10, 0, 35, 25, 10],
            [15, 35, 0, 30, 10],
            [20, 25, 30, 0, 10],
            [10, 10, 10, 10, 0]
        ]
        random.seed(555)
        population = TSPPopulationOrdinalAdjacencyMatrix(2, adjacency_matrix)
        # These tours present [[0, 1, 3, 2, 4],[0, 1, 2, 3, 4]];
        for actual, expected in zip(population.individuals, [[0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]):
            self.assertEqual(actual.genome, expected) # Genome is ordinal tour for individual

    def test_population_fitness_function(self):
        adjacency_matrix = [
            [0, 10, 15, 20, 10],
            [10, 0, 35, 25, 10],
            [15, 35, 0, 30, 10],
            [20, 25, 30, 0, 10],
            [10, 10, 10, 10, 0]
        ]
        random.seed(555)
        population = TSPPopulationOrdinalAdjacencyMatrix(2, adjacency_matrix)
        # These tours present [[0, 1, 3, 2, 4],[0, 1, 2, 3, 4]];
        # With next costs:
        # First: 10+25+30+10=75
        # Second: 10+35+30+10=85
        self.assertEqual(population.fitness_function(population.individuals[0]), 75.0)
        self.assertEqual(population.fitness_function(population.individuals[1]), 85.0)

if __name__ == '__main__':
    unittest.main()
