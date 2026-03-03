"""
Computational Physics
Directory: examples/SeriesExpansions

Code: Interpolate the synethetic data with in-build python splines

Author: Barlev Raymond
"""

# Import
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
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
#print(data.shape)

# Linear splines
splineLinear = interpolate.interp1d(data[0], data[1], kind = 'linear')

# Cubic splines
splineCubic = interpolate.interp1d(data[0], data[1], kind = 'cubic')

# Cubic splines with smoothing
splineSmooth = interpolate.UnivariateSpline(data[0],data[1])
splineSmooth.set_smoothing_factor(500)

# Define a new list to ensure only interpolations
x_list = np.linspace(data[0].min(),data[0].max(),901)

#plot
plt.figure(1)
plt.scatter(data[0], data[1],color='black', label="Randomized function points")
plt.plot(xlist, func(xlist), label="Real function")
plt.plot(x_list, splineLinear(x_list), label="Linear spline")
plt.plot(x_list, splineCubic(x_list), label="Cubic spline")
plt.plot(x_list, splineSmooth(x_list), label="Smooth spline")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()