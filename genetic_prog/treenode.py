# Base class for a tree node
import math


class NodeBase:
    def evaluate(self, variables):
        """Evaluate the node given a dictionary of variable values."""
        raise NotImplementedError("Must be implemented in subclass")

    def __str__(self):
        """Return the string representation of the node."""
        raise NotImplementedError("Must be implemented in subclass")

# Terminal node for variables and constants
class TerminalNode(NodeBase):
    def __init__(self, value):
        self.value = value

    def evaluate(self, variables):
        if isinstance(self.value, str):  # If the value is a variable
            return variables[self.value]
        else:  # If the value is a constant
            return self.value

    def __str__(self):
        return str(self.value)

# Operator node for functions
class OperatorNode(NodeBase):
    def __init__(self, operator, left, right=None):
        self.operator = operator
        self.left = left
        self.right = right

    def evaluate(self, variables):
        """Evaluate the node given a dictionary of variable values."""
        try:
            if self.right is None:
                # Unary operator
                result = self.operator(self.left.evaluate(variables))
            else:
                # Binary operator
                result = self.operator(self.left.evaluate(variables), self.right.evaluate(variables))

            # Проверка на комплексные значения
            if isinstance(result, complex):
                raise ValueError("Evaluation resulted in a complex number")

            # Проверка на переполнение и корректные значения
            MAX_VALUE = 1e10
            MIN_VALUE = -1e10

            return result

        except Exception:
            return float('11111')

    def __str__(self):
        if self.right is None:
            return f"{self.operator.__name__}({self.left})"
        else:
            return f"({self.left} {self.operator.__name__} {self.right})"