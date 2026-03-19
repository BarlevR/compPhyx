"""
Computational Physics
Directory: examples/Derivatives

Code: Velocity and acceleration from displacement data
      Compares Forward, Central and Richardson schemes

Author: Barlev Raymond
"""
import numpy as np
import matplotlib.pyplot as plt
from compPhyx.calculus import ForwardD1, CentralD1, RichardsonD1
from compPhyx.calculus import ForwardD2, CentralD2, RichardsonD2
import compPhyx.logo as logo

print(logo.art)

# Load data
import os
data_path = os.path.join(os.path.dirname(__file__), "velocity_acceleration_data_file.dat")
data = np.loadtxt(data_path)

time = data[:, 0]
disp = data[:, 1]

plt.figure(1)
plt.title("Displacement")
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")
plt.plot(time, disp)

# --- Velocity (1st derivative) ---

vel_forward    = ForwardD1().differentiate(time, disp)
vel_central    = CentralD1().differentiate(time, disp)
vel_richardson = RichardsonD1().differentiate(time, disp)

plt.figure(2)
plt.title("Velocity")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.plot(time, vel_forward,    label="Forward")
plt.plot(time, vel_central,    label="Central")
plt.plot(time, vel_richardson, label="Richardson")
plt.legend()

# --- Acceleration (2nd derivative) ---

accl_forward    = ForwardD2().differentiate(time, disp)
accl_central    = CentralD2().differentiate(time, disp)
accl_richardson = RichardsonD2().differentiate(time, disp)

plt.figure(3)
plt.title("Acceleration")
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s²]")
plt.plot(time, accl_forward,    label="Forward")
plt.plot(time, accl_central,    label="Central")
plt.plot(time, accl_richardson, label="Richardson")
plt.legend()

# --- Max acceleration ---

max_accl = np.max(accl_richardson)
max_time = time[np.argmax(accl_richardson)]
print(f"Max acceleration: {max_accl:.4f} m/s² at t = {max_time:.4f} s")

plt.show()
