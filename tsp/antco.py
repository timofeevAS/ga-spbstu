import random
import numpy as np
import pygame
from sympy.stats.rv import probability

from tsp.drawer.graph_drawer import *
from utils.tsp import TSPData, read_tsp_full_matrix

class Ant:
    def __init__(self, num_cities: int):
        self.num_cities = num_cities
        self.tour = []  # Sequence of visited cities
        self.distance_travelled = 0.0  # Total distance of the tour
        self.current_city = None  # The current city where the ant is located

    def visit_city(self, city: int, distance: float):
        # Add the city to the tour and update the total distance
        self.tour.append(city)
        self.distance_travelled += distance
        self.current_city = city  # Update the current city

    def reset(self, start_city: int):
        # Reset the ant's state for a new iteration and set the starting city
        self.tour = [start_city]
        self.distance_travelled = 0.0
        self.current_city = start_city  # Set the current city to the starting city

class AntColonyTsp:
    def __init__(self,
                 tsp_data: TSPData,
                 num_ants: int,
                 alpha: float,
                 beta: float,
                 evaporation_rate: float,
                 q: float,
                 iterations: int,
                 start_city: int,
                 finish_city: int):
        self.tsp_data = tsp_data
        self.num_ants = num_ants
        self.alpha = alpha  # Importance of pheromone
        self.beta = beta  # Importance of distance
        self.ro = evaporation_rate  # Pheromone evaporation rate
        self.q = q  # Constant for pheromone update
        self.iterations = iterations  # Number of iterations to perform
        self.start_city = start_city
        self.finish_city = finish_city
        # Initialize pheromone levels for all edges with a small constant value
        self.pheromone = [[1.0 for _ in range(tsp_data.dimension)] for _ in range(tsp_data.dimension)]
        self.best_distance = float('inf')  # Track the best distance found
        self.best_solution = []  # Store the best solution (path) found
        # Create a list of ants
        self.ants = [Ant(tsp_data.dimension) for _ in range(num_ants)]
        self.MU_CONST = 1.0
        # Initialize ants
        self.initialize_ants()

    def initialize_ants(self):
        # Initialize each ant to start at the start_city
        for ant in self.ants:
            ant.reset(self.start_city)

    def _select_next_node(self, ant):
        N = self.tsp_data.dimension
        # If the ant has already visited all cities, return it to the start city
        if len(ant.tour) == N:
            # Return to the start city to complete the tour
            start_city = ant.tour[0]
            last_city = ant.current_city
            distance = self.tsp_data.adjacency_matrix[last_city][start_city]
            ant.visit_city(start_city, distance)
            return

        probabilities = []
        i = ant.current_city
        total = 0

        # Calculate the total probability for all unvisited cities
        for j in range(N):
            if j not in ant.tour and self.tsp_data.adjacency_matrix[i][j] > 0:
                tau = (self.pheromone[i][j] ** self.alpha)
                mu = self.MU_CONST / (self.tsp_data.adjacency_matrix[i][j] ** self.beta)
                total += tau * mu

        # Calculate the individual probabilities
        for j in range(N):
            if j not in ant.tour and self.tsp_data.adjacency_matrix[i][j] > 0:
                tau = (self.pheromone[i][j] ** self.alpha)
                mu = self.MU_CONST / (self.tsp_data.adjacency_matrix[i][j] ** self.beta)
                probabilities.append((tau * mu) / total)
            else:
                probabilities.append(0)

        # Choose the next city based on probabilities
        next_city = random.choices(range(N), weights=probabilities, k=1)[0]
        distance = self.tsp_data.adjacency_matrix[i][next_city]
        ant.visit_city(next_city, distance)

    def _balance_pheromone(self):
        N = self.tsp_data.dimension
        # Step 1: Pheromone evaporation
        for i in range(N):
            for j in range(N):
                self.pheromone[i][j] *= (1 - self.ro)  # Reduce pheromone on each edge

        # Step 2: Update pheromone based on the paths taken by the ants
        for ant in self.ants:
            path_length = ant.distance_travelled
            # Update pheromone for all edges traversed by the ant
            for index in range(len(ant.tour) - 1):
                i = ant.tour[index]
                j = ant.tour[index + 1]
                # The amount of pheromone is proportional to the inverse of the path length
                delta_tau = self.q / path_length
                self.pheromone[i][j] += delta_tau
                self.pheromone[j][i] += delta_tau  # Update pheromone in both directions

    def round(self):
        # Step 1: Let the ants build their paths
        for ant in self.ants:
            while len(ant.tour) < self.tsp_data.dimension:
                self._select_next_node(ant)

        # Step 2: Update pheromones
        self._balance_pheromone()

        # Update the best found solution
        for ant in self.ants:
            if ant.distance_travelled < self.best_distance:
                self.best_distance = ant.distance_travelled
                self.best_solution = ant.tour.copy()

        for ant in self.ants:
            ant.reset(self.start_city)

    def run(self):
        pass  # Placeholder for the main algorithm execution

    def run_with_draw(self):
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Ant Visualization")

        twoD_coords = normalize_coordinates_in_center(self.tsp_data.display_data.copy(), WIDTH, HEIGHT, 0.7)
        # Normalize twoD coords

        running = True
        iteration = 0
        while running:
            iteration += 1
            self.round()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill(WHITE)
            draw_tour(screen, twoD_coords, self.best_solution)
            BEST_TOUR_BAYS29 = [0, 27, 5, 11, 8, 4, 25, 28, 2, 1, 19, 9, 3, 14, 17, 16, 13, 21, 10, 18, 24, 6, 22, 26,
                                7, 23,
                                15, 12, 20]

            draw_tour(screen, twoD_coords, BEST_TOUR_BAYS29, GREEN)
            draw_info(screen, {'Current iteration': iteration,
                                    'Best value': self.best_distance})
            draw_ant_graph(screen, twoD_coords, self.tsp_data.adjacency_matrix, self.pheromone)
            pygame.display.flip()

        pygame.quit()

if __name__ == '__main__':
    tspdata = read_tsp_full_matrix('C:\\polytech\\ga-spbstu\\examples\\tsp\\bays29.tsp')
    ac = AntColonyTsp(tspdata, 50, 1, 1, 0.1, 2, 100,
                      0, tspdata.dimension - 1)
    ac.run_with_draw()
