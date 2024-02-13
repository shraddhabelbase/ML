import numpy as np
from typing import List, Tuple

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None

    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it's not a np.ndarray and return. If it can't be cast, raise an error.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert isinstance(x, np.ndarray), "Expected the input to be a list or np.ndarray"
        return x

    def fit(self, x: np.ndarray) -> None:
        x = self._check_is_array(x)
        self.minimum = x.min(axis=0)
        self.maximum = x.max(axis=0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Correctly apply MinMax scaling to the given vector.
        """
        x = self._check_is_array(x)
        # Ensure correct order of operations with parentheses
        return (x - self.minimum) / (self.maximum - self.minimum)

    def fit_transform(self, x: List) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert isinstance(x, np.ndarray), "Expected the input to be a list or np.ndarray"
        return x

    def fit(self, x: np.ndarray) -> None:
        x = self._check_is_array(x)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        # Handle division by zero in std by using np.where
        std_adjusted = np.where(self.std == 0, 1, self.std)
        return (x - self.mean) / std_adjusted

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
