"""
Computational Physics
Directory: src/calculus/integration

Code: Abstract base class for integration schemes

Author: Barlev Raymond
"""
from abc import ABC, abstractmethod


class IntegrationScheme(ABC):
    """
    Abstract base class for numerical integration schemes.

    Subclasses implement __call__(x, y) which integrates
    the discrete data (x, y) over the range [x[0], x[-1]].
    """

    @abstractmethod
    def __call__(self, x, y):
        """
        Numerically integrate y over x.

        Parameters
        ----------
        x : np.ndarray, shape (N,)
            Independent variable (need not be uniform spacing).
        y : np.ndarray, shape (N,)
            Function values at each x.

        Returns
        -------
        float or complex
        """
