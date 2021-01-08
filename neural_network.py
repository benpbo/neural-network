import random
from typing import Tuple
import numpy as np


class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return (a-y)


class NeuralNetwork:
    def __init__(self, *layer_sizes: Tuple[int], cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.large_weight_initializer()
        self.cost = cost

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        """
        self.biases = [np.random.randn(y, 1) for y in self.layer_sizes[1:]]
        self.weights = [
            np.random.randn(x, y)
            for x, y in zip(self.layer_sizes[1:], self.layer_sizes[:-1])
        ]

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """Return the output of the network for 'a' as input."""
        for w, b in zip(self.weights, self.biases):
            a = sigmoid((w @ a) + b)
        return a

    def backprop(self, xs, ys):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        def get_nabla_b(d): return np.add.reduce(d)
        def get_nable_w(d, a): return np.add.reduce(d @ a.transpose((0, 2, 1)))
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = xs
        activations = [xs]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = (w @ activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        delta = None
        for l in range(1, self.num_layers):
            if delta is None:
                delta = (self.cost).delta(zs[-1], activations[-1], ys)
            else:
                delta = (self.weights[-l+1].transpose()
                         @ delta) * sigmoid_prime(zs[-l])
            nabla_b[-l] = get_nabla_b(delta)
            nabla_w[-l] = get_nable_w(delta, activations[-l-1])
        return nabla_b, nabla_w

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        xs, ys = mini_batch
        delta_nabla_b, delta_nabla_w = self.backprop(
            np.array(xs), np.array(ys)
        )
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        batch_length = len(xs)
        self.weights = [w - (eta/batch_length) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/batch_length) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(self.feedforward(x).argmax(), y.argmax())
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        for j in range(epochs):
            random.shuffle(training_data)
            for k in range(0, len(training_data), mini_batch_size):
                mini_batch = zip(*training_data[k:k+mini_batch_size])
                self.update_mini_batch(mini_batch, eta)
            text = f'Epoch {j}: {self.evaluate(test_data)} / {len(test_data)}' if test_data else f'Epoch {j} complete'
            print(text)


# Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
