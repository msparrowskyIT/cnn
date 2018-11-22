import numpy as np
from functools import reduce

class Dense:
    "Dense for 2-D or 3-D numpy arrays."

    def __init__(self, units, use_bias=False, rate=0.5):
        self.units = units
        self.use_bias = use_bias
        self.rate = rate
    
    def train_forward(self, input):
        """Train forward as element wise dot and addition -> dot(weights * input) + bias.
        Cache input for backpropagation."""
        self.forward_input = input
        return self.calculate_forward(input)

    def calculate_forward(self, input):
        "Calculate forward as element wise dot and addition -> dot(weights * input) + bias."
        net = np.dot(self.weights, input)
        if(self.use_bias):
            net += self.bias

        return net

    def __calculate_delta_weights(self, error):
        """Calculate delta weights as sum of element wise multiplication for each pair of error and cached, transponsed input -> error * input.T.
        Then, the sum is divided by number of pairs."""
        if(error.ndim == 2):
            delta_weights = error * self.forward_input.T
        elif(error.ndim == 3):
            delta_weights = np.zeros(self.weights.shape)
            for e,i in zip(error, self.forward_input):
                delta_weights += e * i.T
            delta_weights /= error.shape[0]
            
        return delta_weights

    def __calculate_delta_bias(self, error):
        """Calculate delta bias as sum of error. Then, the sum is divided by number of errors."""
        if(error.ndim == 2):
            return error
        elif(error.ndim == 3):
            error = np.apply_along_axis(np.sum, 0, error)
            return error / error.shape[0]


    def __update(self, error):
        """Update weights and bias using error from follow layer."""
        delta_weights = self.__calculate_delta_weights(error)
        self.weights -= self.rate * delta_weights
        if(self.use_bias):
            delta_bias = self.__calculate_delta_bias(error)
            self.bias -= self.rate * delta_bias
    
    def train_backward(self, error):
        """Update weights and bias and propagare error to previous layer as dot operation -> (weights.T, error)"""
        self.__update(error)

        return np.dot(self.weights.T, error)

    def compile(self, input_size):        
        self.weights =  np.sqrt(2/input_size[0]) * np.random.randn(self.units, input_size[0]) - np.sqrt(1/input_size[0])
        if(self.use_bias):
            self.bias = np.random.randn(self.units, 1)

        return (self.units, 1)