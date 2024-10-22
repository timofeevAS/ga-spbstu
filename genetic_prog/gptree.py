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

    def to_dot(self, filename="tree.dot"):
        """Export the tree to a .dot file for visualization with Graphviz."""
        if not self.root:
            raise ValueError("The tree is empty.")

        # Start building the .dot content
        dot_content = ["digraph G {"]
        self._to_dot_recursive(self.root, dot_content)
        dot_content.append("}")

        # Write to the specified file
        with open(filename, "w") as file:
            file.write("\n".join(dot_content))
        print(f"Dot file saved as {filename}")

    def _to_dot_recursive(self, node, dot_content, parent_id=None):
        """Helper method to recursively build the .dot content."""
        # Generate a unique ID for each node
        node_id = id(node)
        label = str(node.value) if isinstance(node, TerminalNode) else node.operator.__name__

        # Add the current node to the .dot content
        dot_content.append(f'    {node_id} [label="{label}"];')

        # If there's a parent, create an edge from the parent to the current node
        if parent_id is not None:
            dot_content.append(f'    {parent_id} -> {node_id};')

        # Recursively process children if the node is an operator
        if isinstance(node, OperatorNode):
            self._to_dot_recursive(node.left, dot_content, node_id)
            if node.right is not None:
                self._to_dot_recursive(node.right, dot_content, node_id)

    def evaluate(self, variables):
        """Evaluate the tree using the provided variable values."""
        if not self.root:
            raise ValueError("The tree is empty.")
        return self.root.evaluate(variables)

# Example usage:
if __name__ == "__main__":
    tree = GPTree()
    print("Tree structure:")
    tree.pretty_print()
    # Evaluate the tree with given variable values
    variables = {'x1': 2.0, 'x2': 3.0}
    result = tree.evaluate(variables)
    print(f"Evaluation result: {result}")
    # Save the tree as a dot file
    tree.to_dot("tree.dot")
