"""
Computational Physics
Directory: tests/calculus

Code: Fourier transform using the Trapezoidal integration scheme

Author: Barlev Raymond
"""
import numpy as np
import matplotlib.pyplot as plt
from compPhyx.calculus import Trapezoidal
import compPhyx.logo as logo

print(logo.art)

tlist = np.linspace(0, 50, 501)

freq1 = 2.0
freq2 = 0.3
freq3 = 3.5

ylist = 0.5*np.cos(tlist*freq1 + 0.3) + 2.0*np.sin(tlist*freq2) + 1.0*np.cos(tlist*freq3)

scheme = Trapezoidal()

# Single frequency check
omega = 0.3
integrand = 1/np.sqrt(2*np.pi) * ylist * np.exp(1j*omega*tlist)
result = scheme(tlist, integrand)
print(f"Intensity at omega={omega}: {abs(result)**2:.4f}")

# Frequency sweep
omega_list = np.linspace(0, 10, 1001)
ft = np.array([scheme(tlist, 1/np.sqrt(2*np.pi) * ylist * np.exp(1j*om*tlist)) for om in omega_list])

plt.figure(1)
plt.xlabel("Frequency omega")
plt.ylabel("Intensity")
plt.title("Fourier transform via Trapezoidal integration")
plt.plot(omega_list, abs(ft)**2)

# Compare with numpy FFT
ft_fft = np.fft.fft(ylist) / len(ylist)
frequencies = np.arange(len(ylist)) / (len(ylist) * 0.1)

plt.figure(2)
plt.xlabel("Frequency omega")
plt.ylabel("Intensity")
plt.title("Fourier transform via numpy FFT")
plt.xlim([0, 10])
plt.plot(frequencies * 2*np.pi, abs(ft_fft)**2)

plt.show()
