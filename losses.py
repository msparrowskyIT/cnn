import numpy as np

class MeanSquare:
    "Mean square error for 2-D or 3-D numpy arrays."

    def calculate_error(self, target, output):
        "Calculate mean square total error and return derivative as element wise subtraction -(target-output)."
        error = 0.5 * np.power(np.subtract(target, output), 2)
        if(output.ndim == 2):
            print(f"Total mean square error (MSE):\n {np.sum(error)}")
        elif(output.ndim == 3):
            print(f"Total mean square error (MSE) for mini batch:\n {np.sum(np.apply_along_axis(np.sum, 0, error)) / output.shape[0]}")

        return -np.subtract(target, output)


class CrossEntrophy:
    "Cross entrophy error for 2-D or 3-D numpy arrays."
    
    def calculate_error(self, target, output):
        "Calculate cross entrophy total error and return derivative as element wise division -target/output."
        error = -np.multiply(target, np.log(output))
        if(output.ndim == 2):
            print(f"Total cross entrophy error (CEE):\n {np.sum(error)}")
        elif(output.ndim == 3):
            print("Error:\n", error)
            print(f"Total cross entrophy error (CEE) for mini batch:\n {np.sum(np.apply_along_axis(np.sum, 0, error)) / output.shape[0]}")

        return -np.divide(target, output)