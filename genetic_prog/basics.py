# Function set for the operations
import math
import operator

FUNCTION_SET = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    'sin': math.sin,
    'cos': math.cos,
    'exp': math.exp,
    'abs': abs,
    'pow': pow,
}

# Terminal set for variables and constants
TERMINAL_SET = ['x1', 'x2', 1.0, 2.0, 3.0, 4.0]