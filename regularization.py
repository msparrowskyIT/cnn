import numpy as np 

class Dropout:
    "Dropout for 2-D or 3D numpy arrays."

    def __init__(self, rate=0.5):
        self.rate = rate

    def __create_dropout_mask(self, input_shape):
        """Create 2-D or 3-D dropout mask with constant proportion.
        If 2-D, constant proportion on axis 0.
        If 3-D, constant proportion on axis 1."""
        zeros = int(input_shape[-2] * self.rate)
        dropout_mask = np.ones((input_shape[-2], 1))
        dropout_mask[:zeros] = 0
   
        if(len(input_shape) == 2):
            np.random.shuffle(dropout_mask)
        elif(len(input_shape) == 3):
            dropout_mask = np.tile(dropout_mask, (input_shape[0], 1, 1))
            np.apply_along_axis(np.random.shuffle, 1, dropout_mask)

        return dropout_mask

    def train_forward(self, input):
        """Train forward as element wise multiplication dropout_mask * input.
        Cache created dropout mask for backpropagation."""
        self.dropout_mask = self.__create_dropout_mask(input.shape)

        return np.multiply(self.dropout_mask, input)

    def train_backward(self, input):
        """Train forward as element wise multiplication dropout_mask * input.
        Where 'input' is error propagated from followed layer."""
        
        return np.multiply(self.dropout_mask, input)