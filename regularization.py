import numpy as np 

class Dropout:

    def __init__(self, rate=0.5):
        self.rate = rate

    def __create_dropout_mask(self, input_shape):
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
        self.dropout_mask = self.__create_dropout_mask(input.shape)
        print("Dropout \n", self.dropout_mask)
        return np.multiply(input, self.dropout_mask)

    def train_backward(self, input):
        if(self.dropout_mask.ndim == 3):
            self.dropout_mask = np.apply_along_axis(np.any, 0, self.dropout_mask)
        
        return np.multiply(input, self.dropout_mask)