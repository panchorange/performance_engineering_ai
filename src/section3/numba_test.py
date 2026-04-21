from numba import jit
import numpy as np
import time

@jit(nopython=True)
def calculate_sum(arr):
    total = 0
    start
    for i in range(len(arr)):
        total += arr[i]
    return total

# 使用例
data = np.array([1, 2, 3, 4, 5])
start = time.time()
result = calculate_sum(data)
elapsed = time.time() - start
print(f"Result: {result}")
print(f"Elapsed time: {elapsed:.2f} seconds")