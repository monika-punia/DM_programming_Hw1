"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""
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