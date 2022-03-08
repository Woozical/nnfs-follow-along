# import math
import numpy as np
layer_outputs = [
  [4.8, 1.21, 2.385],
  [8.9, -1.81, 0.2],
  [1.41, 1.051, 0.026]
]
# E = math.e

exp_values = np.exp(layer_outputs)
# Axis None = sum of all elements, Axis 0 = sum of cols, Axis 1 = sum of rows
# keepdims, retain shape of input
# I.e keepdims=False: [S1, S2, S3]
#     keepdims=True: [ [S1], [S2], [S3] ]
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(exp_values)
print(norm_values)