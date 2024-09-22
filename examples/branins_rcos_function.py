import math

from core.function import TwoParamFunction


class BraninsRcosFunction(TwoParamFunction):
    def __init__(self):
        a=1
        b=5.1/(4*math.pi**2)
        c=5/math.pi
        d=6
        e=10
        f=1/(8*math.pi)
        function_expr :str = f'{a}*(x2-{b}*x1**2+{c}*x1-{d})**2+{e}*(1-{f})*cos(x1)+{e}'
        param1_name='x1'
        param2_name='x2'
        super().__init__(function_expr, param1_name, param2_name)