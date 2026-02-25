"""
Computational Physics
Directory: examples/SeriesExpansions

Code: Test the calculation of random data from a function

Author: Barlev Raymond
"""

# Import
import numpy as np
import matplotlib.pyplot as plt
from compPhyx.dataGenerator import generate_synthetic_data
import compPhyx.logo as logo

# Function
def func(x):
    return 15 + 2.4*x - 0.5*x**2 - 0.35*x**3

# Data
npoints = 21
xlist = np.linspace(-5, 5, npoints)
data = generate_synthetic_data(func(xlist), xlist, npoints)

print(logo.art)
#print(data)

#plot
plt.figure(1)
plt.plot(xlist, func(xlist), label="Real function")
plt.scatter(data[0], data[1], label="Randomized function points")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
