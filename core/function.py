import sympy as sp


class OneParamFunction:
    def __init__(self, function_expr: str, param_name : str):
        self.function_expr = function_expr
        self.param_name = sp.symbols(param_name)

        try:
            self.function_expr = sp.sympify(function_expr)
        except sp.SympifyError:
            raise ValueError(f"Incorrect math expression: {function_expr}")

    def evaluate(self, value: float) -> float:
        """
        Evaluate function value by parameter

        :param value: Value input to function
        :return: f(value).
        """
        return float(self.function_expr.subs(self.param_name, value))

    def __str__(self) -> str:
        return f"f({self.param_name}) = {self.function_expr}"


class TwoParamFunction:
    def __init__(self, function_expr: str, param1_name : str, param2_name : str):
        self.function_expr = function_expr
        self.param1_name = sp.symbols(param1_name)
        self.param2_name = sp.symbols(param2_name)

        try:
            self.function_expr = sp.sympify(function_expr)
        except sp.SympifyError:
            raise ValueError(f"Incorrect math expression: {function_expr}")

    def evaluate(self, value1: float, value2: float) -> float:
        """
        Evaluate function value by parameter

        :param value1: Value1 input to function
        :param value2: Value1 input to function
        :return: f(value).
        """
        return float(self.function_expr.subs({self.param1_name: value1, self.param2_name: value2}))

    def __str__(self) -> str:
        return f"f({self.param1_name}, {self.param2_name}) = {self.function_expr}"