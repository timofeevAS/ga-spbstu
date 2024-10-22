import random
from treenode import TerminalNode, OperatorNode
from basics import FUNCTION_SET, TERMINAL_SET

class GPTree:
    def __init__(self, root=None):
        # Initialize the tree with a root node or generate a random tree
        self.root = root if root is not None else self.generate_random_tree()

    def generate_random_tree(self, depth=3):
        """Recursively generates a random tree."""
        if depth == 0 or (depth > 1 and random.random() < 0.3):
            # Create a terminal node (leaf)
            terminal = random.choice(TERMINAL_SET)
            return TerminalNode(terminal)
        else:
            # Create an operator node
            function = random.choice(list(FUNCTION_SET.keys()))
            if function in ['sin', 'cos', 'exp', 'abs']:  # Unary operators
                left = self.generate_random_tree(depth - 1)
                return OperatorNode(FUNCTION_SET[function], left)
            else:  # Binary operators
                left = self.generate_random_tree(depth - 1)
                right = self.generate_random_tree(depth - 1)
                return OperatorNode(FUNCTION_SET[function], left, right)

    def pretty_print(self, node=None, indent=0):
        """Recursively prints the tree structure with indentation."""
        if node is None:
            node = self.root
        prefix = ' ' * (indent * 4)
        if isinstance(node, TerminalNode):
            print(f"{prefix}{node.value}")
        elif isinstance(node, OperatorNode):
            print(f"{prefix}{node.operator.__name__}")
            self.pretty_print(node.left, indent + 1)
            if node.right is not None:
                self.pretty_print(node.right, indent + 1)

# Example usage:
if __name__ == "__main__":
    tree = GPTree()
    print("Tree structure:")
    tree.pretty_print()
