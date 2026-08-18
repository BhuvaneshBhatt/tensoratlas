from tensoratlas import ScalarField, VectorField, coordinate_chart, gradient, divergence, curl, laplacian
import sympy as sp


def main():
    chart = coordinate_chart('Euclidean', 'Cartesian', 3)
    x, y, z = chart.symbols()
    f = ScalarField(chart, x**2 + y**2 + z**2)
    v = VectorField(chart, sp.Matrix([[y], [-x], [z]]))
    gradient(f)
    divergence(v)
    curl(v)
    laplacian(f)


if __name__ == '__main__':
    main()
