"""
Computational Physics
Directory: src/approx/fitData

Code: Polynomial fitting via gradient descent

Author: Barlev Raymond
"""
import numpy as np


class PolyFit:
    """
    Fits a polynomial of given degree to (X, Y) data using gradient descent.

    Parameters
    ----------
    degree : int
        Degree of the polynomial (number of coefficients = degree + 1).
    iterations : int
        Number of gradient descent steps (default: 10000).
    learning_rate : float
        Step size for gradient descent (default: 1e-5).
    """

    def __init__(self, degree, iterations=10000, learning_rate=1e-5):
        if degree < 0:
            raise ValueError("degree must be >= 0")
        self.degree = degree
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.coefficients = None

    def _poly(self, x, a):
        return sum(a[k] * x**k for k in range(len(a)))

    def _error(self, a, data):
        return float(np.sum((data[1] - self._poly(data[0], a))**2))

    def _gradient(self, a, data):
        residuals = data[1] - self._poly(data[0], a)
        return -2 * np.array([
            np.sum(residuals * data[0]**k)
            for k in range(len(a))
        ])

    def fit(self, data):
        """
        Fit the polynomial to data.

        Parameters
        ----------
        data : np.ndarray, shape (2, N)
            data[0] = x values, data[1] = y values

        Returns
        -------
        self
        """
        a = 2 * np.random.rand(self.degree + 1) - 1
        for _ in range(self.iterations):
            a = a - self.learning_rate * self._gradient(a, data)
        self.coefficients = a
        return self

    def evaluate(self, x):
        """Evaluate the fitted polynomial at x (scalar or array)."""
        if self.coefficients is None:
            raise RuntimeError("Call fit() before evaluate()")
        return self._poly(x, self.coefficients)

    def error(self, data):
        """Sum of squared residuals for the fitted polynomial."""
        if self.coefficients is None:
            raise RuntimeError("Call fit() before error()")
        return self._error(self.coefficients, data)
