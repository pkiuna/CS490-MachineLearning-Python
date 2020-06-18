import numpy as np

# Getting 15 random values from 1 to 20
x = np.random.randint(1,20,15)
print("Input:")
print(x)

# Replace maximum value in numpy to 0
x[np.where(x == np.amax(x))] = 0
print("Output:")
print(x)