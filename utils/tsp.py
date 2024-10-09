from typing import List, Optional, Tuple


class TSPData:
    def __init__(self, name: str, tsp_type: str, comment: Optional[str], dimension: int,
                 edge_weight_type: str, edge_weight_format: str, display_data_type: str,
                 adjacency_matrix: List[List[int]], display_data: List[Tuple[float, float]]):
        self.name = name  # Name of the TSP instance
        self.tsp_type = tsp_type  # Type of the problem (TSP)
        self.comment = comment  # Optional comment about the problem
        self.dimension = dimension  # Number of cities (dimension of the problem)
        self.edge_weight_type = edge_weight_type  # The type of edge weight (e.g., EXPLICIT)
        self.edge_weight_format = edge_weight_format  # Format of the edge weight matrix (e.g., FULL_MATRIX)
        self.display_data_type = display_data_type  # Data display type (e.g., TWOD_DISPLAY)
        self.adjacency_matrix = adjacency_matrix  # The full matrix containing distances between cities
        self.display_data = display_data  # Coordinates (display data) of each city


def read_tsp_full_matrix(filepath: str) -> TSPData:
    with open(filepath, 'r') as file:
        name = ''
        tsp_type = ''
        comment = None
        dimension = 0
        edge_weight_type = ''
        edge_weight_format = ''
        display_data_type = ''
        adjacency_matrix = []
        display_data = []
        is_matrix_section = False
        is_display_data_section = False

        for line in file:
            line = line.strip()

            # Parse the name of the TSP data
            if line.startswith("NAME:"):
                name = line.split(":")[1].strip()

            # Parse the type of the problem
            elif line.startswith("TYPE:"):
                tsp_type = line.split(":")[1].strip()

            # Parse the comment (if available)
            elif line.startswith("COMMENT:"):
                comment = line.split(":")[1].strip()

            # Parse the dimension of the TSP (number of cities)
            elif line.startswith("DIMENSION:"):
                dimension = int(line.split(":")[1].strip())

            # Parse the edge weight type (e.g., EXPLICIT)
            elif line.startswith("EDGE_WEIGHT_TYPE:"):
                edge_weight_type = line.split(":")[1].strip()

            # Parse the edge weight format (e.g., FULL_MATRIX)
            elif line.startswith("EDGE_WEIGHT_FORMAT:"):
                edge_weight_format = line.split(":")[1].strip()

            # Parse the display data type
            elif line.startswith("DISPLAY_DATA_TYPE:"):
                display_data_type = line.split(":")[1].strip()

            # Detect the start of the adjacency matrix
            elif line.startswith("EDGE_WEIGHT_SECTION"):
                is_matrix_section = True
                is_display_data_section = False

            # Parse the full matrix section
            elif is_matrix_section and line != "DISPLAY_DATA_SECTION":
                row = list(map(int, line.split()))
                adjacency_matrix.append(row)

            # Detect the start of the display data section
            elif line.startswith("DISPLAY_DATA_SECTION"):
                is_display_data_section = True
                is_matrix_section = False

            # Parse the display data (coordinates of the cities)
            elif is_display_data_section and line != "EOF":
                _, x, y = map(float, line.split())
                display_data.append((x, y))

            # Stop parsing when reaching the end of the file
            elif line == "EOF":
                break

    # Return the parsed TSP data as an object
    return TSPData(name=name, tsp_type=tsp_type, comment=comment, dimension=dimension,
                   edge_weight_type=edge_weight_type, edge_weight_format=edge_weight_format,
                   display_data_type=display_data_type, adjacency_matrix=adjacency_matrix,
                   display_data=display_data)


def generate_ordinal_tour(route: List[int], original_order: List[int]) -> List[int]:
    ordinal_tour = []
    remaining_nodes = original_order.copy()

    for node in route:
        index = remaining_nodes.index(node)
        ordinal_tour.append(index)

        remaining_nodes.pop(index)

    return ordinal_tour