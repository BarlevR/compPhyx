"""
Computational Physics
Directory: examples/

Code: Polynomial fitting example using PolyFit

Author: Barlev Raymond
"""
import numpy as np
import matplotlib.pyplot as plt
from compPhyx.dataGenerator import generate_synthetic_data
from compPhyx.approx import PolyFit
import compPhyx.logo as logo


def func(x):
    return 15 + 2.4*x - 0.5*x**2 - 0.35*x**3


npoints = 21
xlist = np.linspace(-5, 5, npoints)
data = generate_synthetic_data(func(xlist), xlist, npoints)

print(logo.art)

fitter = PolyFit(degree=3, iterations=10000, learning_rate=1e-5)
fitter.fit(data)

print("Fitted coefficients:", fitter.coefficients)
print("Fit error:", fitter.error(data))

plt.scatter(data[0], data[1], label="Noisy data")
plt.plot(xlist, fitter.evaluate(xlist), 'black', label="Fitted polynomial")
plt.plot(xlist, func(xlist), 'orange', label="True function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
