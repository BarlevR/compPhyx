"""
Computational Physics
Directory: examples/Integrals/RotationalMotion

Code: Rotational kinetic energy of a solid sphere rotating around z-axis
      Uses 3D sum rule over discrete grid points inside the sphere.
      Analytical: I = 2/5 * m * r^2

Author: Barlev Raymond
"""
import numpy as np
import matplotlib.pyplot as plt
import compPhyx.logo as logo

print(logo.art)

m = 1.0     # kg
r = 1.0     # m
omega = 1.0  # rad/s

analytical = 2/5 * m * r**2 / 2 * omega**2
print(f"Analytical: {analytical:.6f} J")

num_points = 30
coord_list = []
counter = 0
contribution = 0.0

for x in np.linspace(-r, r, num_points):
    for y in np.linspace(-r, r, num_points):
        for z in np.linspace(-r, r, num_points):
            if np.linalg.norm([x, y, z]) <= r:
                coord_list.append([x, y, z])
                counter += 1
                contribution += np.linalg.norm([x, y, 0])**2

coord_list = np.transpose(coord_list)
E = (m / counter) * contribution / 2 * omega**2
print(f"Numerical:  {E:.6f} J")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.scatter(coord_list[0], coord_list[1], coord_list[2], s=0.1)
ax.set_title("Solid sphere")
plt.show()
