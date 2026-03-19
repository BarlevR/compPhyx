"""
Computational Physics
Directory: examples/test

Code: Test derivative schemes against analytical derivatives

Author: Barlev Raymond
"""
import numpy as np
import matplotlib.pyplot as plt
from compPhyx.calculus import ForwardD1, BackwardD1, CentralD1, RichardsonD1
from compPhyx.calculus import ForwardD2, BackwardD2, CentralD2, RichardsonD2
import compPhyx.logo as logo

print(logo.art)

def f(x):
    return np.sin(x) * x - 1/100 * x**3

def df(x):
    return np.cos(x) * x + np.sin(x) - 3/100 * x**2

def d2f(x):
    return -np.sin(x) * x + 2*np.cos(x) - 6/100 * x

xlist = np.linspace(-10, 10, 2001)
h = xlist[1] - xlist[0]

# --- First derivatives ---

schemes_d1 = {
    "Forward":    ForwardD1(h),
    "Backward":   BackwardD1(h),
    "Central":    CentralD1(h),
    "Richardson": RichardsonD1(h),
}

analytical_d1 = df(xlist)

plt.figure(1)
plt.title("First derivative — schemes vs analytical")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.plot(xlist, analytical_d1, 'black', label="Analytical")
for name, scheme in schemes_d1.items():
    plt.plot(xlist, scheme(f, xlist), '--', label=name)
plt.legend()

plt.figure(2)
plt.title("First derivative — errors")
plt.xlabel("x")
plt.ylabel("Error")
for name, scheme in schemes_d1.items():
    plt.plot(xlist, analytical_d1 - scheme(f, xlist), label=name)
plt.legend()

# --- Second derivatives ---

schemes_d2 = {
    "Forward":    ForwardD2(h),
    "Backward":   BackwardD2(h),
    "Central":    CentralD2(h),
    "Richardson": RichardsonD2(h),
}

analytical_d2 = d2f(xlist)

plt.figure(3)
plt.title("Second derivative — schemes vs analytical")
plt.xlabel("x")
plt.ylabel("f''(x)")
plt.plot(xlist, analytical_d2, 'black', label="Analytical")
for name, scheme in schemes_d2.items():
    plt.plot(xlist, scheme(f, xlist), '--', label=name)
plt.legend()

plt.figure(4)
plt.title("Second derivative — errors")
plt.xlabel("x")
plt.ylabel("Error")
for name, scheme in schemes_d2.items():
    plt.plot(xlist, analytical_d2 - scheme(f, xlist), label=name)
plt.legend()

plt.show()
