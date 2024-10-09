import random
from typing import List, Tuple

from core.individual import Individual
from core.population import Population
from utils.tsp import generate_ordinal_tour


class OrdinalTourIndividual(Individual):
    def __init__(self, genome: List[int], ordinal: List[int]):
        super().__init__(genome)
        self.ordinal = ordinal

    def mutate(self) -> None:
        # Two random indexes.
        idx1, idx2 = random.sample(range(1, len(self.genome) - 2), 2)

        # Swap.
        self.genome[idx1], self.genome[idx2] = self.genome[idx2], self.genome[idx1]

    def crossover(self, other: "OrdinalTourIndividual") -> Tuple["OrdinalTourIndividual", "OrdinalTourIndividual"]:
        if len(other.genome) != len(self.genome):
            raise ValueError(f"Length of genome must be equals: {len(other.genome)} != {len(self.genome)}")

        crossover_idx = random.randint(0, len(self.genome) - 1) # Zero based index

        child1 = OrdinalTourIndividual(self.genome[:crossover_idx] + other.genome[crossover_idx:], self.ordinal)
        child2 = OrdinalTourIndividual(other.genome[:crossover_idx] + self.genome[crossover_idx:], self.ordinal)

        return child1, child2

    def __str__(self):
        return f'{self.genome}'

    def __repr__(self):
        return f'{self.genome}'



    def get_tour(self) -> List[int]:
        tour: List[int] = []
        ordinal_copy = self.ordinal.copy()

        # Iterate each index in genome.
        for ordinal_idx in self.genome:
            tour.append(ordinal_copy.pop(ordinal_idx))

        return tour


class TSPPopulationOrdinalAdjacencyMatrix(Population):
    def __init__(self, population_size: int, adjacency_matrix: List[List[float]]):
        super().__init__(population_size)
        self.individuals: List[OrdinalTourIndividual] = [] # Override typehint

        self.adjacency_matrix = adjacency_matrix
        self.adjacency_list = self.adjacency_list_from_matrix()
        self.node_count = len(self.adjacency_matrix)  # Count of nodes.
        self.ordinal = list(range(self.node_count))
        self.init_population()



    def adjacency_list_from_matrix(self) -> List[List[int]]:
        adjacency_list = [[] for _ in range(len(self.adjacency_matrix))]

        for i in range(len(self.adjacency_matrix)):
            for j in range(len(self.adjacency_matrix[i])):
                if self.adjacency_matrix[i][j] != 0:
                    adjacency_list[i].append(j)

        return adjacency_list

    def init_population(self) -> None:
        n = self.node_count
        for i in range(self.population_size):
            tour: List[int] = [0]  # Tour begins at node 0
            current_node = 0
            visited = {0, n - 1}  # Track visited nodes, starting with node 0

            # Generate a random tour by visiting each node exactly once
            for _ in range(n - 2):  # Exclude starting node (0) and final node (n-1)
                neighbors = [node for node in self.adjacency_list[current_node] if node not in visited]
                if not neighbors:
                    break  # Fail-safe to prevent getting stuck without available nodes
                next_node = random.choice(neighbors)  # Randomly choose the next unvisited node
                tour.append(next_node)
                visited.add(next_node)
                current_node = next_node

            # Ensure the tour ends at the last node (n-1)
            tour.append(n - 1)

            # Add the new individual to the population
            self.individuals.append(OrdinalTourIndividual(generate_ordinal_tour(tour, self.ordinal), self.ordinal))

    def fitness_function(self, individual: OrdinalTourIndividual) -> float:
        tour_cost = 0.
        tour = individual.get_tour()

        current_node = tour.pop(0) # Start with first node of tour.
        while tour:
            next_node = tour.pop()
            tour_cost += self.adjacency_matrix[current_node][next_node]
            current_node = next_node

        return tour_cost