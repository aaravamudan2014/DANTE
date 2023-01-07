# importing all memory kernels
from utils.MemoryKernel import *
# import all point processes
from point_processes.PointProcessCollection import *
# import all visualization tools
from utils.VisualizationUtility import generatePP_plot
from point_processes.NonParametricEstimator import *
# still dependant on matlab to show the final plot
import matplotlib.pyplot as plt


def test_poisson():
    """[summary]
    """
    sourceNames = ['ownEvents']  # this TPP only uses its own events as source.
    mk = ConstantMemoryKernel()

    truetpp = PoissonTPP(
        mk, sourceNames, desc='Poisson TPP with a Gamma Gompertz (1.0, 2.0) kernel')
    T = 100
    numRealizations = 10
    Realizations = []

    for r in range(0, numRealizations):
        # Realization is an np array of time points, the last point is the right-censoring time
        Realization = truetpp.simulate(T, [np.array([])])
        Realizations.append([Realization])

    maxTime = max(Realization)

    # generate am inhomogeneous poisson process with spline kernel generated
    # from nelson aalen estimates (nelson aalen estimates from realizations)
    spline_tpp = generateSplineKernelProcess(Realizations, maxTime)
    realization = spline_tpp.simulate(T, [np.array([])])

    generatePP_plot(realization, spline_tpp)
    plt.show()


def test_hawkes():
    """[summary]
    """
    pass


def test_split_pop():
    """[summary]
    """
    pass


def test_non_parametric():
    """[summary]
    """
    pass


def main():
    pass


if __name__ == "__main__":
    main()
