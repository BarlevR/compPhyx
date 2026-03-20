"""
Computational Physics
Directory: examples/Integrals/MagneticFieldWire

Code: Magnetic vector potential and field of a long straight wire
      Integration over wire length using sum rule.

Author: Barlev Raymond
"""
import numpy as np
import matplotlib.pyplot as plt
import compPhyx.logo as logo

print(logo.art)

mu0 = 1
j0 = 1       # A/m^2
r0 = 0.001   # m
l0 = 1000    # m

def j(r):
    return np.array([0.0, 0.0, j0])

coordMax = 4.9
numpoints = 50
d = 2 * coordMax / (numpoints - 1)

coords = np.array(np.meshgrid(
    np.linspace(-coordMax, coordMax, numpoints),
    np.linspace(-coordMax, coordMax, numpoints),
    np.zeros(1),
    indexing='ij'
))

A = np.array(np.meshgrid(
    np.zeros(numpoints),
    np.zeros(numpoints),
    np.zeros(1),
    indexing='ij'
))

numint = 5001
dz = (2 * l0) / (numint - 1)
df = np.pi * r0**2

for ix in np.arange(numpoints):
    for iy in np.arange(numpoints):
        r = np.array([-coordMax + ix*d, -coordMax + iy*d, 0.0])
        for zj in np.linspace(-l0, l0, numint):
            rj = np.array([0.0, 0.0, zj])
            A[:, ix, iy, 0] += j(rj) / np.sqrt(r[0]**2 + r[1]**2 + rj[2]**2)

A = A * mu0 / (4 * np.pi) * df * dz

# Vector potential
plt.figure(1)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.contourf(coords[0, :, :, 0], coords[1, :, :, 0], A[2, :, :, 0])
plt.colorbar(label="Vector potential Az")

# Analytical comparison
xlist = np.linspace(-coordMax, coordMax, 10001)
plt.figure(2)
plt.xlabel("x [m]")
plt.ylabel("Az")
plt.scatter(coords[0, :, numpoints//2, 0], A[2, :, numpoints//2, 0], label="Numerical")
plt.plot(
    xlist,
    mu0/(2*np.pi) * j0 * df * np.log(2*l0 / np.sqrt(xlist**2 + coords[1, 0, numpoints//2, 0]**2)),
    'red', label="Analytical"
)
plt.legend()

# Magnetic field via central differences on A
B = np.array(np.meshgrid(
    np.zeros(numpoints),
    np.zeros(numpoints),
    np.zeros(1),
    indexing='ij'
))

B[0, 1:-1, 1:-1, 0] =  (A[2, 1:-1, 2:, 0]  - A[2, 1:-1, :-2, 0]) / (2*d)
B[1, 1:-1, 1:-1, 0] = -(A[2, 2:, 1:-1, 0]  - A[2, :-2, 1:-1, 0]) / (2*d)
B[2, 1:-1, 1:-1, 0] =  (A[1, 2:, 1:-1, 0]  - A[1, :-2, 1:-1, 0]) / (2*d) \
                      - (A[0, 1:-1, 2:, 0]  - A[0, 1:-1, :-2, 0]) / (2*d)

plt.figure(3)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.contourf(coords[0, :, :, 0], coords[1, :, :, 0], np.sqrt(B[0,:,:,0]**2 + B[1,:,:,0]**2 + B[2,:,:,0]**2))
plt.colorbar(label="|B|")

plt.figure(4)
ax = plt.axes(projection='3d')
ax.axis(False)
scale = 2e5
ax.quiver(coords[0], coords[1], coords[2], B[0]*scale, B[1]*scale, B[2]*scale)
ax.set_title("Magnetic field B")

plt.show()
