import sys
import numpy as np

sys.path.insert(0, "..\\assignment1")
from linear_classifer import softmax_with_cross_entropy, l2_regularization

class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.clean_gradients()

    def clean_gradients(self):
        self.grad = np.zeros_like(self.value)

class ReLULayer:
    def forward(self, X):
        result = X.copy()
        result[result <= 0] = 0
        self.x_greater = result > 0
        return result

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """

        d_result = np.zeros_like(d_out)
        d_result[self.x_greater] = d_out[self.x_greater]

        return d_result

    def clean_gradients(self):
        return

    def params(self):
        # ReLU Doesn't have any parameters
        return {}

class FullyConnectedLayer:
    def __init__(self, n_input, n_output, reg=0):
        self.W = Param(0.001 * np.random.randn(n_input + 1, n_output))
        self.reg = reg
        self.X = None

    def compute_loss_and_gradients(self, X, y):
        """
        Behaves like fully functional model.
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """

        self.clean_gradients()
        predictions = self.forward(X)
        predictions_count = predictions.shape[0]

        reg_loss, dreg_loss = l2_regularization(self.W.value, self.reg)
        self.W.grad += dreg_loss/predictions_count

        loss, grad = softmax_with_cross_entropy(predictions, y)
        loss += reg_loss
        loss/=predictions_count
        grad/=predictions_count

        self.backward(grad)

        return loss

    def predict(self, X):
        """
        Produces layer predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """

        return np.argmax(self.forward(X), axis=1)

    def forward(self, X):
        self.X = np.hstack((
            X,
            np.ones((X.shape[0], 1))
        ))

        return self.X@self.W.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """

        self.W.grad += self.X.T@d_out
        
        d_result = d_out@(self.W.value.T)
        last_column_idx = d_result.shape[1] - 1
        return d_result[:, :last_column_idx]

    def clean_gradients(self):
        self.W.clean_gradients()

    def params(self):
        return {
            'W': self.W
        }
