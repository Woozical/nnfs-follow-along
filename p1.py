import sys
import numpy as np
import matplotlib

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
# Input Sample A, B, C
inputs = [
  [1, 2, 3, 2.5],
  [2.0, 5.0, -1.0, 2.0],
  [-1.5, 2.7, 3.3, -0.8]
]
###

# Layer 1, Neurons 1, 2, 3
weights = [
  [0.2, 0.8, -0.5, 1.0],
  [0.5, -0.91, 0.26, -0.5],
  [-0.26, -0.27, 0.17, 0.87]
]
biases = [2, 3, 0.5]
###

# Layer 2, Neurons 4, 5, 6
weights2 = [
  [0.1, -0.14, 0.5],
  [-0.5, 0.12, -0.33],
  [-0.44, 0.73, -0.13]
]
biases2 = [-1, 2, -0.5]


l1_output = np.dot(inputs, np.array(weights).T) + biases
print(l1_output)
# Output structure:
# [
#   Sample A: [N1 Output, N2 Output, N3 Output]
#   Sample B: [N1 Output, N2 Output, N3 Output]
#   Sample C: [N1 Output, N2 Output, N3 Output]
# ]

# We can't perform the inverse operation np.dot(weights, np.array(inputs).T)
# because then each bias would correspond with a sample set, rather than a neuron
# I.E. 
# Output structure:
# [
#   Neuron 1 Outputs: [Sample A Result + N1 Bias, Sample B Result + N2 Bias, Sample C Result + N3 Bias]
#   Neuron 2 Outputs: [Sample A Result + N1 Bias, Sample B Result + N2 Bias, Sample C Result + N3 Bias]
#   Neuron 3 Outputs: [Sample A Result + N1 Bias, Sample B Result + N2 Bias, Sample C Result + N3 Bias]
# ]

### Forward feeding to layer 2
l2_output = np.dot(l1_output, np.array(weights2).T) + biases2
print(l2_output)

# layer_outputs = []
# for neuron_weights, neuron_bias in zip(weights, biases):
#   neuron_output = 0
#   for n_input, weight in zip(inputs, neuron_weights):
#     neuron_output += n_input * weight
  
#   neuron_output += neuron_bias
#   layer_outputs.append(neuron_output)
# print(layer_outputs)