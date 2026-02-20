"""
Computational Physics
Directory: examples/SeriesExpansions

Code: Test the numerical derivative calculation
User provides a function and a value at which the derivative is calculated

Author: Barlev Raymond
"""
# Imports
import src.logo as logo
import numpy as np
from src.calculus import nth_derivative

# Provide function
def func(x):
    return 2*np.sin(x)**2 + x

# Point at which to calculate derivative
x0 = 10.5

# Step size
h = 0.1

# Order of derivative
n = 2

# Function call
derivative = nth_derivative(func, x0,h, n)

# Outputs
print(logo.art)
print(f"The value of the function at {x0} is {func(x0)}")
print(f"The {n}th derivative at {x0} is {derivative}")
