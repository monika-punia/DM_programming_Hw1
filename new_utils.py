"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""

from numpy.typing import NDArray
import numpy as np


def scale_data(data):
    # Scale between 0 and 1
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    if data.size == 0:
        raise ValueError("Input array is empty.")

    if not np.issubdtype(data.dtype, np.float64) and not np.issubdtype(data.dtype, np.float32):
        raise ValueError("Input array must have a floating-point data type.")

    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input array contains NaN or infinite values.")
"""
    data = (data - data.min()) / (data.max() - data.min())
    return data

def scale_data_1(y_bi = NDArray[np.int32]):
    # Check if the elements in y are integers or not
    if not issubclass(y_bi.dtype.type, np.int32):
        return False
    
    return True

def check_labels(y):
    """
    Check if labels are integers.
    Parameters:
        y: Labels
    Returns:
        Boolean indicating whether labels are integers
    """
    # Test that the 1D y array are all integers
    return np.issubdtype(y.dtype, np.int32)



def filter_9s_convert_to_01(X, y, frac):

    # Count the number of 9s in the array
    num_9s = np.sum(y == 9)

    # Calculate the number of 9s to remove (90% of the total number of 9s)
    num_9s_to_remove = int(frac * num_9s)

    # Identifying indices of 9s in y
    indices_of_9s = np.where(y == 9)[0]

    num_9s_to_remove = int(np.ceil(len(indices_of_9s) * frac))
    indices_to_remove = np.random.choice(
        indices_of_9s, size=num_9s_to_remove, replace=False
    )

    # Removing the selected indices from X and y
    X = np.delete(X, indices_to_remove, axis=0)
    y = np.delete(y, indices_to_remove)

    y[y == 7] = 0
    y[y == 9] = 1
    return X, y

def filter_imbalanced_7_9s(X, y):
    # Filter out only 7s and 9s
    seven_nine_idx = (y == 7) | (y == 9)
    X_binary = X[seven_nine_idx]
    y_binary = y[seven_nine_idx]

    # Convert 7s to 0s and 9s to 1s
    y_binary = np.where(y_binary == 7, 0, 1)

    # Remove 90% of 9s
    nines_idx = np.where(y_binary == 1)[0]
    remove_n = int(len(nines_idx) * 0.9)  # 90% to remove
    np.random.shuffle(nines_idx)
    remove_idx = nines_idx[:remove_n]

    # Keep only the desired indices
    keep_idx = np.setdiff1d(np.arange(len(X_binary)), remove_idx)
    X_imbalanced = X_binary[keep_idx]
    y_imbalanced = y_binary[keep_idx]

    return X_imbalanced, y_imbalanced
    