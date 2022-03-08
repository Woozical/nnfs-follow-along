import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class LayerDense:
  
  def __init__(self, n_inputs, n_neurons):
    # Initialize randomized weights for each connection between input and this layer's neurons
    self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
    # Initialize a bias of 0 for each neuron in layer
    self.biases = np.zeros((1, n_neurons))

  def __len__(self):
    # Return the # of neurons in this layer
    return np.shape(self.weights)[1]

  def forward(self, inputs):
    # Save output from given inputs to layer's output property
    self.output = np.dot(inputs, self.weights) + self.biases

# Activation functions
class Activation:
  @classmethod
  def reLU(cls, inputs):
    """ Rectified Linear Activation Function """
    return np.maximum(0, inputs)

  @classmethod
  def softmax(cls, inputs):
    """ Softmax Activation Function """
    # Subtract each input by the max of each input batch
    # Prior to exponentiation, our inputs will be bounded between -Infinity and 0
    # After exponentiation, our inputs will be bounded between 0 and 1
    # This will prevent potential overflow errors from exponentiation of our inputs
    # After normalization, output will be identical compared to if we didnt subtract max
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities

X, y = spiral_data(samples=100, classes=3)

dense1 = LayerDense(2, 3)
dense1.forward(X)
activation1 = Activation.reLU(dense1.output)

dense2 = LayerDense(3, 3)
dense2.forward(activation1)
activation2 = Activation.softmax(dense2.output)

# print(activation2[:5])
print(X)