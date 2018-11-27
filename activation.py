import numpy as np 

class Activation:
    pass

class Sigmoid(Activation):
    """Sigmoid for stochastic gradient descent (2-D numpy array) or mini-batch gradient descent (3D numpy array)."""
    
    def __calculate_derivative(self):
        """Calculate derivative as element wise operation: output * (1-output)"""
        
        return self.output * (1 - self.output)

    def forward(self, input):
        """Train forward as element wise operation: e^input / (1 + e^input)."""
        sigmoid = np.vectorize(lambda i: 1 / (1 + np.exp(-i)) if i > 0 else np.exp(i) / (1 + np.exp(i)))
        
        return sigmoid(input)
    
    def train_forward(self, input):
        """Train forward as element wise operation: e^input / (1 + e^input).
        The method caches output for backpropagation."""
        self.output = self.forward(input)
        
        return self.output 

    def train_backward(self, input):
        """Train backward as element wise operation: input * output * (1 - output)."""        
        
        return input * self.__calculate_derivative()

class Tanh(Activation):
    """Hyperbolic tangent for stochastic gradient descent (2-D numpy array) or mini-batch gradient descent (3D numpy array)."""
    
    def __calculate_derivative(self):
        """Calculate derivative as element wise operation: output * (1-output)"""
        
        return 1 - self.output**2

    def forward(self, input):
        """Train forward as element wise operation: tanh(input)."""
        
        return np.tanh(input)
    
    def train_forward(self, input):
        """Train forward as element wise operation: e^input / (1 + e^input).
        The method caches output for backpropagation."""
        self.output = self.forward(input)
        
        return self.output 

    def train_backward(self, input):
        """Train backward as element wise operation: input * output * (1 - output)."""        
        
        return input * self.__calculate_derivative()


class Relu(Activation):
    """Rectifier for stochastic gradient descent (2-D numpy array) or mini-batch gradient descent (3D numpy array)."""

    def __calculate_derivative(self):
        """Calculate derivative as element wise operation: 0 if output <= 0 else 1"""

        return np.where(self.output <= 0, 0, 1)

    def forward(self, input):
        """Train forward as element wise operation: e^input / (1 + e^input)."""
        
        return np.maximum(input, 0)

    def train_forward(self, input):
        """Train forward as element wise operation: 0 if output <= 0 else 1.
        The method caches output for backpropagation."""
        self.output = self.forward(input)
        
        return self.output

    def train_backward(self, input):
        """Train backward as element wise operation: input * (0 if output <= 0 else 1)."""        

        return input * self.__calculate_derivative()

class Softmax(Activation):
    """Softmax for stochastic gradient descent (2-D numpy array) or mini-batch gradient descent (3D numpy array)."""
    
    def __calculate_derivative(self):
        """Calculate derivative as element wise operation: output * (1-output)"""
        
        return self.output * (1 - self.output)

    def forward(self, input):
        """Train forward as element wise operation: e^input / Σe^input."""
        if(input.ndim == 2):
            input -= np.max(input)
            return np.exp(input) / np.sum(np.exp(input))
        elif(input.ndim == 3):
            input = np.apply_along_axis(lambda i: i - np.max(i), 1, input)
            return np.apply_along_axis(lambda i: np.exp(i) / np.sum(np.exp(i)), 1, input)
    
    def train_forward(self, input):
        """Train forward as element wise operation: e^input / Σe^input.
        The method caches output for backpropagation."""
        self.output = self.forward(input)
        
        return self.output
        
    def train_backward(self, input):
        """Train backward as element wise operation: input * output * (1 - output)."""        
        
        return input * self.__calculate_derivative()