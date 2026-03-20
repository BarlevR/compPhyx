"""
Computational Physics
Directory: examples/Integrals/RotationalMotion

Code: Rotational kinetic energy of a hollow spherical shell rotating around z-axis
      Uses 3D sum rule over discrete grid points between inner and outer radius.
      Analytical: I = 2/5 * m * (r1^5 - r2^5) / (r1^3 - r2^3)

Author: Barlev Raymond
"""
import numpy as np
import matplotlib.pyplot as plt
import compPhyx.logo as logo

print(logo.art)

m = 1.0     # kg
r1 = 1.0    # outer radius, m
r2 = 0.8    # inner radius, m
omega = 1.0  # rad/s

analytical = 0.5 * omega**2 * 2/5 * m * (r1**5 - r2**5) / (r1**3 - r2**3)
print(f"Analytical: {analytical:.6f} J")

num_points = 40
coord_list = []
counter = 0
contribution = 0.0

for x in np.linspace(-r1, r1, num_points):
    for y in np.linspace(-r1, r1, num_points):
        for z in np.linspace(-r1, r1, num_points):
            if r2 <= np.linalg.norm([x, y, z]) <= r1:
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
ax.set_title("Hollow spherical shell")
plt.show()
