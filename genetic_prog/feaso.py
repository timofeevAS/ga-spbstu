import numpy as np
import matplotlib.pyplot as plt
import math
from treenode import TerminalNode, OperatorNode
from basics import FUNCTION_SET
from pgtree import PGTree

FEASO_RANGE_X1 = (-5, 5)
FEASO_RANGE_X2 = (-5, 5)

def fEaso():
    """Create a GPTree representing the function fEaso."""
    # Create the terminal nodes for the variables and constants
    x1 = TerminalNode('x1')
    x2 = TerminalNode('x2')
    pi = TerminalNode(math.pi)

    # Create the expression (x1 - pi)
    x1_minus_pi = OperatorNode(FUNCTION_SET['-'], x1, pi)
    # Create the expression (x2 - pi)
    x2_minus_pi = OperatorNode(FUNCTION_SET['-'], x2, pi)

    # Create the squared terms: (x1 - pi)^2 and (x2 - pi)^2
    x1_minus_pi_squared = OperatorNode(FUNCTION_SET['pow'], x1_minus_pi, TerminalNode(2))
    x2_minus_pi_squared = OperatorNode(FUNCTION_SET['pow'], x2_minus_pi, TerminalNode(2))

    # Create the sum of squares: (x1 - pi)^2 + (x2 - pi)^2
    sum_of_squares = OperatorNode(FUNCTION_SET['+'], x1_minus_pi_squared, x2_minus_pi_squared)

    # Apply the negation: -((x1 - pi)^2 + (x2 - pi)^2)
    negated_sum = OperatorNode(FUNCTION_SET['*'], TerminalNode(-1), sum_of_squares)

    # Create the exponential part: exp(-((x1 - pi)^2 + (x2 - pi)^2))
    exp_term = OperatorNode(FUNCTION_SET['exp'], negated_sum)

    # Create the cosine terms: cos(x1) and cos(x2)
    cos_x1 = OperatorNode(FUNCTION_SET['cos'], x1)
    cos_x2 = OperatorNode(FUNCTION_SET['cos'], x2)

    # Create the product of the cosine terms: cos(x1) * cos(x2)
    cos_product = OperatorNode(FUNCTION_SET['*'], cos_x1, cos_x2)

    # Create the final multiplication: -cos(x1) * cos(x2) * exp(...)
    final_product = OperatorNode(FUNCTION_SET['*'], cos_product, exp_term)

    # Apply the negation: - (cos(x1) * cos(x2) * exp(...))
    negation = OperatorNode(FUNCTION_SET['*'], TerminalNode(-1), final_product)

    # Create the GPTree with the negation as the root
    tree = PGTree(negation)
    return tree


def plot_feaso():
    """Plot the fEaso function over a 3D surface using matplotlib."""
    # Create the GPTree for the fEaso function
    tree = fEaso()

    # Generate a grid of values for x1 and x2 in the range -100 to 100
    x1_values = np.linspace(*FEASO_RANGE_X1, num=100)
    x2_values = np.linspace(*FEASO_RANGE_X2, num=100)
    X1, X2 = np.meshgrid(x1_values, x2_values)

    # Compute the function values on the grid
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            variables = {'x1': X1[i, j], 'x2': X2[i, j]}
            Z[i, j] = tree.evaluate(variables)

    # Plot the surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis')

    # Add labels and title
    ax.set_title('3D Surface plot of fEaso(x1, x2)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('fEaso(x1, x2)')

    # Show the plot
    plt.show()


def plot_f_x1_x2_pgtree(tree: PGTree):
    """Plot the plot with tree (x1,x2)"""
    # Generate a grid of values for x1 and x2 in the range -100 to 100
    x1_values = np.linspace(*FEASO_RANGE_X1, num=100)
    x2_values = np.linspace(*FEASO_RANGE_X2, num=100)
    X1, X2 = np.meshgrid(x1_values, x2_values)

    # Compute the function values on the grid
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            variables = {'x1': X1[i, j], 'x2': X2[i, j]}
            Z[i, j] = tree.evaluate(variables)

    # Plot the surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis')

    # Add labels and title
    ax.set_title('3D Surface plot of tree(x1, x2)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('tree(x1, x2)')

    # Show the plot
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Plot the fEaso function
    plot_feaso()
    fEaso().to_dot('feaso.dot')
    print(fEaso().get_depth())
    print(f'min: {fEaso().evaluate({"x1":math.pi, "x2":math.pi})}')