import sys
import numpy as np

from layers import FullyConnectedLayer, ReLULayer

sys.path.insert(0, "..\\assignment1")
from linear_classifer import softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """

        self.n_input = n_input
        self.n_output = n_output
        self.hidden_layer_size = hidden_layer_size
        self.reg = reg
        self.layers = None

    def ensure_layers(self):
        if self.layers is None:
            self.layers = [
                ("Input Layer", FullyConnectedLayer(self.n_input, self.hidden_layer_size)),
                ("ReLU Layer", ReLULayer()),
                ("Hidden Layer", FullyConnectedLayer(self.hidden_layer_size, self.n_output)),
            ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """

        self.ensure_layers()

        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for _,layer in self.layers:
            layer.clean_gradients()

        predictions = X
        predictions_count = predictions.shape[0]
        full_reg_loss = 0
        for _,layer in self.layers:
            predictions = layer.forward(predictions)
            for param in layer.params().values():
                reg_loss, dreg_loss = l2_regularization(param.value, self.reg)
                full_reg_loss += reg_loss
                param.grad += dreg_loss/predictions_count

        loss, grad = softmax_with_cross_entropy(predictions, y)
        loss += full_reg_loss
        loss/=predictions_count
        grad/=predictions_count

        for _,layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """

        predictions = X

        if self.layers is None:
            raise BaseException("Model must be fitted before predictin values")

        for _,layer in self.layers:
            predictions = layer.forward(predictions)
        return np.argmax(predictions, axis=1)

    def params(self):
        result = {}

        self.ensure_layers()

        for name,layer in self.layers:
            layer_params = layer.params()
            for key in layer_params.keys():
                result[name + "_" + key] = layer_params[key]
        return result
