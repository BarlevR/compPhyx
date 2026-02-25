"""
Computational Physics
Directory: DataGenerator/syntheticData_XY.py

Code: Generate randomized synthetic data (X,Y)
User can provide a function to which random noise can be added

Author: Barlev Raymond
"""

# Import
import numpy as np
import random

def generate_synthetic_data(f, x, num_points, key=666):
    # f: Function values to modify
    # x: the data range X given by the user
    # num_points: Number of points in the dataset we need
    # key: random key (optional)

    random.seed(key)
    data = np.array([x + 0.25*(2*np.random.rand(num_points)-1), f + 5.0*(2*np.random.rand(num_points)-1)])
    return data
