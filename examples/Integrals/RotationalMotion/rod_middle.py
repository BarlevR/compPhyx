"""
Computational Physics
Directory: examples/Integrals/RotationalMotion

Code: Rotational kinetic energy of a rod rotating around its middle
      E = omega^2/2 * (m/s) * integral_-s/2^s/2 (r^2 dr)
      Analytical: I = 1/12 * m * s^2

Author: Barlev Raymond
"""
import numpy as np
import matplotlib.pyplot as plt
from compPhyx.calculus import Trapezoidal, Simpson
import compPhyx.logo as logo

print(logo.art)

m = 1.0     # kg
s = 1.0     # m
omega = 1.0  # rad/s

num_points = 1001  # odd for Simpson
r_list = np.linspace(-s/2, s/2, num_points)

analytical = 1/12 * m * s**2 / 2 * omega**2
print(f"Analytical:   {analytical:.6f} J")

for name, scheme in [("Trapezoidal", Trapezoidal()), ("Simpson", Simpson())]:
    I = (m / s) * scheme(r_list, r_list**2)
    E = omega**2 / 2 * I
    print(f"{name:12s}: {E:.6f} J")

plt.plot(r_list, np.zeros(num_points), label="Rod")
plt.xlim([-0.6*s, 0.6*s])
plt.ylim([-0.5, 0.5])
plt.xlabel("r [m]")
plt.title("Rod rotating around its middle")
plt.show()
