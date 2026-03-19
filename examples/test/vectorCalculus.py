"""
Computational Physics
Directory: examples/Derivatives

Code: Gradient, divergence and curl of multidimensional functions

Author: Barlev Raymond
"""
import numpy as np
import matplotlib.pyplot as plt
from compPhyx.calculus import gradient, divergence, curl
import compPhyx.logo as logo

print(logo.art)

# --- Scalar field for gradient ---
# f(x,y,z) = exp(-x^2 - y^4)

def f(r):
    return np.exp(-r[0]**2 - r[1]**4)

r = np.array([0.5, -1.2, -8.0])

grad = gradient(f, r)
print("Gradient of f at r:", grad)

# Analytical solution
analytical_grad = np.array([
    -2*r[0] * np.exp(-r[0]**2 - r[1]**4),
    -4*r[1]**3 * np.exp(-r[0]**2 - r[1]**4),
    0.0
])
print("Analytical gradient:  ", analytical_grad)
print("Gradient error:       ", np.abs(grad - analytical_grad))

# --- Vector field for divergence and curl ---
# g(r) = r / |r|  (unit radial field)

def g(r):
    return r / np.linalg.norm(r)

div = divergence(g, r)
print("\nDivergence of g at r:", div)
print("Analytical divergence:", 2 / np.linalg.norm(r))

curl_vec = curl(g, r)
print("\nCurl of g at r:", curl_vec)
print("Analytical curl: [0, 0, 0] (irrotational field)")

# --- Plot scalar field ---

x2, y2 = np.meshgrid(np.linspace(-2, 2, 201), np.linspace(-2, 2, 201))
z2 = f(np.array([x2, y2, np.zeros_like(x2)]))

plt.figure(1)
ax = plt.axes(projection='3d')
ax.contour3D(x2, y2, z2, 50)
ax.set_title("Scalar field f(x,y)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f")

# --- Plot vector field ---

x3, y3, z3 = np.meshgrid(np.linspace(-2, 2, 6), np.linspace(-2, 2, 6), np.linspace(-2, 2, 6))
values = g(np.array([x3, y3, z3]))

plt.figure(2)
ax2 = plt.axes(projection='3d')
ax2.set_title("Vector field g(r) = r / |r|")
ax2.axis(False)
scale = 5
ax2.quiver(x3, y3, z3, values[0]*scale, values[1]*scale, values[2]*scale)

plt.show()
