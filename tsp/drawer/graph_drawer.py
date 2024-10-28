import pygame

WIDTH, HEIGHT = 1000, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
PHEROMONE_COLOR = pygame.color.Color(125,0,225,50)


def normalize_coordinates_in_center(twoD_coords, width, height, scale_factor=1.0):
    x_coords = [x for x, y in twoD_coords]
    y_coords = [y for x, y in twoD_coords]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Normalizing coordinates
    normalized_coords = []
    for x, y in twoD_coords:
        norm_x = (x - min_x) / (max_x - min_x) * width * scale_factor
        norm_y = (y - min_y) / (max_y - min_y) * height * scale_factor
        normalized_coords.append((norm_x, norm_y))

    # Find center of normalized coord.
    center_x = sum(x for x, y in normalized_coords) / len(normalized_coords)
    center_y = sum(y for x, y in normalized_coords) / len(normalized_coords)

    # Centerize.
    offset_x = width / 2 - center_x
    offset_y = height / 2 - center_y

    centered_coords = [(x + offset_x, y + offset_y) for x, y in normalized_coords]

    return centered_coords

def draw_graph(screen, coordinates, adjacency_matrix):
    num_vertices = len(coordinates)
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if adjacency_matrix[i][j] > 0:
                pygame.draw.line(screen, BLACK, coordinates[i], coordinates[j], 2)

    for coord in coordinates:
        pygame.draw.circle(screen, RED, coord, 10)

    pygame.draw.circle(screen, GREEN, coordinates[0], 10)
    pygame.draw.circle(screen, GREEN, coordinates[-1], 10)


def draw_ant_graph(screen, coordinates, adjacency_matrix, pheromones):
    num_vertices = len(coordinates)

    # Create a transparent surface with the same size as the screen
    transparent_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if adjacency_matrix[i][j] > 0:
                # Draw on the transparent surface
                pygame.draw.line(transparent_surface, PHEROMONE_COLOR,
                                 coordinates[i], coordinates[j],
                                 int(30 * pheromones[i][j]))

    # Blit the transparent surface onto the main screen
    screen.blit(transparent_surface, (0, 0))

    for coord in coordinates:
        pygame.draw.circle(screen, RED, coord, 10)

    pygame.draw.circle(screen, GREEN, coordinates[0], 10)

def draw_tour(screen, coordinates, tour, color=BLUE):
    if len(tour) < 2:
        return

    for i in range(len(tour) - 1):
        start_index = tour[i]
        end_index = tour[i + 1]

        start_pos = coordinates[start_index]
        end_pos = coordinates[end_index]

        pygame.draw.line(screen, color, start_pos, end_pos, 2)

    start_pos = coordinates[tour[-1]]  # последняя вершина
    end_pos = coordinates[tour[0]]  # первая вершина
    pygame.draw.line(screen, color, start_pos, end_pos, 4)

def draw_info(screen, info):
    font = pygame.font.SysFont(None, 24)

    x_offset = screen.get_width() - 10
    y_offset = 10

    for key, value in info.items():
        text = f"{key}: {value}"
        text_surface = font.render(text, True, (0, 0, 0))  # Чёрный цвет
        text_rect = text_surface.get_rect()
        text_rect.topright = (x_offset, y_offset)
        screen.blit(text_surface, text_rect)
        y_offset += text_rect.height + 5
