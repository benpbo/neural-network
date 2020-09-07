from neural_network import NeuralNetwork
import numpy as np


with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']
    validation_images = data['validation_images']
    validation_labels = data['validation_labels']


training_data = list(zip(training_images, training_labels))
test_data = list(zip(test_images, test_labels))
validation_data = (validation_images, validation_labels)


net = NeuralNetwork(28**2, 15, 10)
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
