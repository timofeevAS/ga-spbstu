import unittest
import random
from pgtree import PGTree
from treenode import TerminalNode, OperatorNode
from basics import FUNCTION_SET, TERMINAL_SET

class TestPGTree(unittest.TestCase):

    def setUp(self):
        """Set up a simple tree for testing."""
        # Seed for reproducibility
        random.seed(42)
        # Create a sample tree
        self.tree = PGTree()
        self.tree.root = OperatorNode(FUNCTION_SET['+'],
                                      TerminalNode('x1'),
                                      OperatorNode(FUNCTION_SET['*'],
                                                   TerminalNode('x2'),
                                                   TerminalNode(3)))

    def test_copy(self):
        """Test the copy method to ensure deep copying."""
        tree_copy = self.tree.copy()
        # Check that the root and structure are equal but not the same object
        self.assertNotEqual(id(self.tree.root), id(tree_copy.root))
        self.assertEqual(self.tree.root.operator, tree_copy.root.operator)
        self.assertEqual(self.tree.root.left.value, tree_copy.root.left.value)
        self.assertEqual(self.tree.root.right.operator, tree_copy.root.right.operator)
        # Modify the copy and check that the original does not change
        tree_copy.root.left = TerminalNode('x3')
        self.assertNotEqual(self.tree.root.left.value, tree_copy.root.left.value)

    def test_get_random_node(self):
        """Test if get_random_node returns a valid node."""
        random_node = self.tree.get_random_node()
        # Ensure the node is part of the tree
        all_nodes = self.tree._collect_nodes(self.tree.root)
        self.assertIn(random_node, all_nodes)

    def test_swap_subtrees(self):
        """Test if swap_subtrees correctly swaps subtrees."""
        node1 = self.tree.root.left
        node2 = self.tree.root.right
        # Perform the swap
        self.tree.swap_subtrees(node1, node2)
        self.assertEqual(str(self.tree.root), '((x2 mul 3) add x1)')

    def test_evaluate(self):
        """Test if the evaluate method computes the correct value."""
        variables = {'x1': 2.0, 'x2': 3.0}
        result = self.tree.evaluate(variables)
        # The original tree is: (x1 + (x2 * 3)), with x1=2 and x2=3
        expected_result = 2 + (3 * 3)
        self.assertAlmostEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
