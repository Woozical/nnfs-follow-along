import numpy as np

np.random.seed(0)

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

# Rectified Linear Activation (raw)
for i in inputs:
  output.append(max(0,i))

class Layer_Dense:
  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))
  
  def __len__(self):
    return np.shape(self.weights)[1]
  
  def forward(self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases

# Rectified Linear Activation Function
class Activation_ReLU:
  @classmethod
  def forward(cls, inputs):
    return np.maximum(0, inputs)


print(output)