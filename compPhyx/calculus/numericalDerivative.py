"""
Computational Physics
Directory: src/calculus
Code: Numerically calculate (higher) derivatives
Approximate the nth derivative of f at x0 using an (n+1)-point forward difference:
        f^(n)(x0) ≈ (1/h^n) * sum_{k=0..n} (-1)^(k+n) * C(n,k) * f(x0 + k h) * n!

Author: Barlev Raymond
"""

# import libraries
import math

# Define function
def nth_derivative(f, x, h, n):
    # f: Function
    # x: Argument of f
    # h: Step size
    # n: nth derivative

    total = 0
    factorial = math.factorial(n)
    for k in range(n+1):
        coeff = (-1)**(k+n) * factorial / (math.factorial(k) * math.factorial(n-k))
        total +=  coeff* f(x + k*h)

    return total / h**n
