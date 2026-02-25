"""
Computational Physics
Directory: examples/SeriesExpansions

Code: Taylor Expansion of a General function (using TaylorSeries class)

Author: Barlev Raymond
"""

import numpy as np
import matplotlib.pyplot as plt
import compPhyx.logo as logo

# Adjust this import to match your package name + file location
# Example assumes: compPhyx/approx/taylor.py contains class TaylorSeries
from compPhyx.approx.taylorExpansion import TaylorSeries


def func(x):
    #return 2*np.sin(x)**2 + x
    return 2 * np.sin(x) * np.cos(x / 2)


def main():
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim([-2, 2])

    xlist = np.linspace(-5, 5, 101)
    plt.scatter(xlist, func(xlist), label="f(x)", s=12)

    nmax = 5
    h = 0.01

    # Taylor expansions about different x0
    ts0 = TaylorSeries(func, x0=0.0, nmax=nmax, h=h)
    ts2 = TaylorSeries(func, x0=2.0, nmax=nmax, h=h)
    tsm3 = TaylorSeries(func, x0=-3.0, nmax=nmax, h=h)

    plt.plot(xlist, ts0.evaluate(xlist), label="Taylor about x0=0", linewidth=2)
    plt.plot(xlist, ts2.evaluate(xlist), label="Taylor about x0=2", linewidth=2)
    plt.plot(xlist, tsm3.evaluate(xlist), label="Taylor about x0=-3", linewidth=2)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    print(logo.art)
    main()
