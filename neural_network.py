import numpy as np
import random


class NeuralNetwork:
    def __init__(self, *layer_sizes: list):
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
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [
            np.random.randn(x, y) for x, y in zip(layer_sizes[1:], layer_sizes[:-1])
        ]

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """Return the output of the network for 'a' as input."""
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w @ a + b)
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
        delta = self.cost_derivative(activations[-1], ys) * sigmoid_prime(zs[-1])
        nabla_b[-1] = get_nabla_b(delta)
        nabla_w[-1] = get_nable_w(delta, activations[-2])
        for l in range(2, self.num_layers):
            delta = (self.weights[-l+1].transpose() @ delta) * sigmoid_prime(zs[-l])
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
            np.array(xs), np.array(ys))
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        batch_length = len(xs)
        self.weights = [
            w-(eta/batch_length)*nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b-(eta/batch_length)*nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                zip(*training_data[k:k+mini_batch_size])
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(self.feedforward(x).argmax(), y.argmax())
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives for the output activations."""
        return output_activations-y


# Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
