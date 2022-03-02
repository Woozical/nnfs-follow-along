import numpy as np

np.random.seed(0)

### Input samples
X = [
  [1.0, 2.0, 3.0, 2.5],
  [2.0, 5.0, -1.0, 2.0],
  [-1.5, 2.7, 3.3, -0.8]
]

class Layer_Dense:
  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))

  ## Returns the # of neurons in this layer
  def __len__(self):
    return np.shape(self.weights)[1]
  
  def forward(self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(len(X[0]), 5)
layer2 = Layer_Dense(len(layer1), 2)

layer1.forward(X)
print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)