import numpy as np 

class Dropout:
    "Dropout for stochastic gradient descent (2-D numpy array) or mini-batch gradient descent (3D numpy array)."

    def __init__(self, rate=0.5):
        self.rate = rate

    def __create_dropout_mask(self, input_shape):
        """Create 2-D or 3-D dropout mask with constant proportion.
        If 2-D, constant proportion on axis 0.
        If 3-D, constant proportion on axis 1."""
        zeros = int(self.rate * input_shape[-2])
        dropout_mask = np.ones((input_shape[-2], 1))
        dropout_mask[:zeros] = 0
   
        if(len(input_shape) == 2):
            np.random.shuffle(dropout_mask)
        elif(len(input_shape) == 3):
            dropout_mask = np.tile(dropout_mask, (input_shape[0], 1, 1))
            np.apply_along_axis(np.random.shuffle, 1, dropout_mask)

        return dropout_mask

    def train_forward(self, input):
        """Train forward as element wise multiplication: dropout_mask * input.
        The method caches created dropout mask for backpropagation."""
        self.dropout_mask = self.__create_dropout_mask(input.shape)

        return self.dropout_mask * input

    def train_backward(self, input):
        """Train forward as element wise multiplication: dropout_mask * input."""
                
        return self.dropout_mask * input