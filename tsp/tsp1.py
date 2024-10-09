import random
from collections import deque
from typing import List, Tuple

from core.ga import GeneticAlgorithm
from core.individual import Individual
from core.population import Population
from utils.tsp import generate_ordinal_tour, read_tsp_full_matrix


class OrdinalTourIndividual(Individual):
    def __init__(self, genome: List[int], ordinal: List[int]):
        super().__init__(genome)
        self.ordinal = ordinal

    def mutate(self) -> None:
        # Two random indexes.
        # Mutation of "phenotype".
        tour = self.get_tour()
        idx1, idx2 = random.sample(range(1, len(tour) - 2), 2)

        # Swap.
        tour[idx1], tour[idx2] = tour[idx2], tour[idx1]

        self.genome = generate_ordinal_tour(tour, self.ordinal)

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

    def copy(self) -> "TSPPopulationOrdinalAdjacencyMatrix":
        return TSPPopulationOrdinalAdjacencyMatrix(self.population_size, self.adjacency_matrix)

class TSPGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, population_size: int, tsp_file_path: str, crossover_p: float, mutation_p: float):
        super().__init__(population_size, crossover_p, mutation_p)
        self.tsp_data = read_tsp_full_matrix(tsp_file_path)
        self.population = TSPPopulationOrdinalAdjacencyMatrix(self.population_size,
                                                              self.tsp_data.adjacency_matrix)


    def round(self) -> None:
        # Any round of genetic algorithm starts with:
        # 1. Selection.
        # 2. Reproduction.
        # 3. Mutation.
        # 4. Reduction (to initial size).

        # 1. Selection:
        self.selection(3, 2)

        # 2. Reproduction.
        self.reproduction()

        # 3. Mutation.
        self.mutation()

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

        for i in range(self.population_size):
            if random.random() <= self.crossover_p:
                parent1, parent2 = random.choices(self.population.individuals, k=2)

                c1, c2 = parent1.crossover(parent2)

                # Mutation for childs
                if random.random() <= self.mutation_p:
                    c1.mutate()

                if random.random() <= self.mutation_p:
                    c2.mutate()
                childs.append(c1)
                childs.append(c2)


        p.individuals.extend(childs)


    def mutation(self) -> None:
        # In this case we implement mutation inside reproduction ONLY for childs.
        pass

    def reduction(self) -> None:
        p = self.population
        p.individuals.sort(key=lambda x: p.fitness_function(x))
        p.individuals = p.individuals[:p.population_size]

    def run(self, iter_count: int) -> None:
        for i in range(iter_count):
            self.round()

if __name__ == '__main__':
    ga = TSPGeneticAlgorithm(10, '../examples/tsp/bays29.tsp', 0.6, 0.1)
    ga.run(100)

    for individ in ga.population.individuals[:5]:
        print(f'{individ}: {ga.population.fitness_function(individ)}')