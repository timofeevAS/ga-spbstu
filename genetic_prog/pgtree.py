import random
from treenode import TerminalNode, OperatorNode
from basics import FUNCTION_SET, TERMINAL_SET

class PGTree:
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

    def copy(self):
        """Create a deep copy of the tree."""
        return PGTree(self._copy_recursive(self.root))

    def _copy_recursive(self, node):
        """Helper method to recursively copy a tree."""
        if isinstance(node, TerminalNode):
            return TerminalNode(node.value)
        elif isinstance(node, OperatorNode):
            left_copy = self._copy_recursive(node.left)
            right_copy = self._copy_recursive(node.right) if node.right else None
            return OperatorNode(node.operator, left_copy, right_copy)
        return None

    def get_random_node(self):
        """Select a random node from the tree."""
        # Collect all nodes in the tree
        nodes = self._collect_nodes(self.root)
        # Return a random node if the list is not empty
        return random.choice(nodes) if nodes else None

    def _collect_nodes(self, node):
        """Helper method to collect all nodes in the tree."""
        nodes = [node]
        if isinstance(node, OperatorNode):
            # Recursively collect nodes from the left and right children
            nodes.extend(self._collect_nodes(node.left))
            if node.right is not None:
                nodes.extend(self._collect_nodes(node.right))
        return nodes

    def swap_subtrees(self, node1, node2):
        """Swap the subtrees rooted at node1 and node2 by exchanging their instances."""
        if node1 is None or node2 is None:
            return

        # Create copies of the subtrees rooted at node1 and node2
        node1_copy = self._copy_recursive(node1)
        node2_copy = self._copy_recursive(node2)

        # Replace node1 with node2's copy and node2 with node1's copy
        self._replace_node(node1, node2_copy)
        self._replace_node(node2, node1_copy)

    def _replace_node(self, target_node, new_subtree):
        """Helper method to replace target_node with new_subtree in the tree."""
        if self.root is target_node:
            # If the target node is the root, replace the root directly
            self.root = new_subtree
        else:
            # Otherwise, perform a recursive search to replace the node
            self._replace_node_recursive(self.root, target_node, new_subtree)

    def _replace_node_recursive(self, current_node, target_node, new_subtree):
        """Recursively search for the target_node and replace it with new_subtree."""
        if current_node is None:
            return False

        if isinstance(current_node, OperatorNode):
            if current_node.left is target_node:
                current_node.left = new_subtree
                return True
            elif current_node.right is target_node:
                current_node.right = new_subtree
                return True

            # Recursively search in children
            replaced_in_left = self._replace_node_recursive(current_node.left, target_node, new_subtree)
            replaced_in_right = self._replace_node_recursive(current_node.right, target_node,
                                                             new_subtree) if current_node.right else False
            return replaced_in_left or replaced_in_right

        return False

    def is_correct(self) -> bool:
        """Check if the tree represents a valid mathematical expression."""
        if not self.root:
            return False  # An empty tree is not a valid expression
        return self._is_correct_recursive(self.root)

    def _is_correct_recursive(self, node) -> bool:
        """Helper method to recursively check the correctness of the tree."""
        if isinstance(node, TerminalNode):
            # A terminal node is always correct
            return True
        elif isinstance(node, OperatorNode):
            # Check for unary operators
            if node.operator in [FUNCTION_SET['sin'], FUNCTION_SET['cos'], FUNCTION_SET['exp'], FUNCTION_SET['abs']]:
                # Unary operators should have only one child (left)
                if node.left is None or node.right is not None:
                    return False
                # Recursively check the left child
                return self._is_correct_recursive(node.left)
            else:
                # Binary operators should have both children (left and right)
                if node.left is None or node.right is None:
                    return False
                # Recursively check both children
                return self._is_correct_recursive(node.left) and self._is_correct_recursive(node.right)
        # If it's neither a TerminalNode nor an OperatorNode, it's incorrect
        return False

# Example usage:
if __name__ == "__main__":
    tree = PGTree()
    print("Tree structure:")
    tree.pretty_print()
    # Evaluate the tree with given variable values
    variables = {'x1': 2.0, 'x2': 3.0}
    result = tree.evaluate(variables)
    print(f"Evaluation result: {result}")
    # Save the tree as a dot file
    tree.to_dot("tree.dot")
