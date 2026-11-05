## is the neural net

import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, genome=None):
        self.layer_sizes = layer_sizes

        # Total number of weights + biases
        self.genome_length = sum(
            (layer_sizes[i] + 1) * layer_sizes[i + 1]
            for i in range(len(layer_sizes) - 1)
        )

        if genome is None:
            genome = np.random.uniform(-1, 1, self.genome_length)
        elif len(genome) != self.genome_length:
            raise ValueError(f"Genome must have length {self.genome_length}")

        self.genome = np.array(genome, dtype=float)
        self._unpack_genome()

    def _unpack_genome(self):
        """Turn the flat genome into weight matrices and bias vectors."""
        self.weights, self.biases = [], []
        idx = 0
        for i in range(len(self.layer_sizes) - 1):
            n_in, n_out = self.layer_sizes[i], self.layer_sizes[i + 1]
            w_end = idx + n_in * n_out
            self.weights.append(self.genome[idx:w_end].reshape(n_in, n_out))
            idx = w_end
            self.biases.append(self.genome[idx:idx + n_out])
            idx += n_out
            
    
    
    
    def evaluate(self, inputs):
        """Forward pass: returns the network's output for the given inputs."""
        x = np.array(inputs, dtype=float)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            # tanh on hidden layers, linear on the output layer
            if i < len(self.weights) - 1:
                x = np.tanh(x)
        return x













