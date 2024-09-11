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