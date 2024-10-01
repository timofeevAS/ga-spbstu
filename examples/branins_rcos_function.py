import math
import numpy as np
import matplotlib.pyplot as plt

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

    def print_plot_with_points(self, dots):
        # Example of plot:
        num: int = 50
        x1 = np.linspace(-5, 10, num, dtype=float)
        x2 = np.linspace(0, 15, num, dtype=float)

        X1, X2 = np.meshgrid(x1, x2)

        f = BraninsRcosFunction()
        Z = np.array([[f.evaluate(X1[i][j], X2[i][j]) for j in range(num)] for i in range(num)])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(X1, X2, Z, cmap='hsv', alpha=0.5)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('f(X1, X2)')
        ax.set_title('Branin RCos Function')

        # Plot dots on the surface
        print(dots)
        if dots:
            dots_x1, dots_x2 = zip(*dots)
            dots_z = [f.evaluate(dot_x1, dot_x2) for dot_x1, dot_x2 in dots]

            dots_z = list(map(lambda x: x + 5, dots_z))
            ax.scatter(dots_x1, dots_x2, dots_z, color='black', s=50, label='Points', marker='P')

        plt.show()

if __name__ == '__main__':
    BraninsRcosFunction().print_plot_with_points([(-math.pi, 12.275), (math.pi,2.275), (3*math.pi, 2.475)])