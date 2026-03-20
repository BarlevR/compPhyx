"""
Computational Physics
Directory: examples/Integrals/RotationalMotion

Code: Rotational kinetic energy of a single point mass

Author: Barlev Raymond
"""
import numpy as np
import matplotlib.pyplot as plt
import compPhyx.logo as logo

print(logo.art)

m = 1.0    # kg
r = 1.0    # m
omega = 1.0  # rad/s

t_array = np.linspace(0, 2*np.pi, 100)
x_array = r * np.cos(omega * t_array)
y_array = r * np.sin(omega * t_array)

E = 0.5 * m * (r * omega)**2
print(f"Rotational energy: {E:.6f} J")
print(f"Analytical:        {0.5 * m * omega**2 * r**2:.6f} J")

fig, ax = plt.subplots()
ax.set_aspect(1)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.scatter(x_array, y_array)
ax.set_title("Point mass circular motion")
plt.show()
