"""
Computational Physics
Directory: tests/calculus

Code: Test integration schemes against analytical results

Author: Barlev Raymond
"""
import numpy as np
import compPhyx.logo as logo
from compPhyx.calculus import Sum, Trapezoidal, Simpson

print(logo.art)

def func(x):
    return 0.5 + 0.1*x + 0.2*x**2 + 0.03*x**3

def analytical(a, b):
    def F(x):
        return 0.5*x + 0.1/2*x**2 + 0.2/3*x**3 + 0.03/4*x**4
    return F(b) - F(a)

a, b = -3.0, 3.0
x = np.linspace(a, b, 13)
y = func(x)

exact = analytical(a, b)
print(f"Analytical:   {exact:.6f}")
print()

for name, scheme in [("Sum", Sum()), ("Trapezoidal", Trapezoidal()), ("Simpson", Simpson())]:
    result = scheme(x, y)
    error = abs(result - exact)
    print(f"{name:12s}: {result:.6f}  (error: {error:.2e})")
