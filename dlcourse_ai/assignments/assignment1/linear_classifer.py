import numpy as np

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''

    maxes = np.max(predictions, axis=1)[:, np.newaxis]
    predictions_normalized = predictions - maxes
    predictions_exp = np.exp(predictions_normalized)
    predictions_sum = np.sum(predictions_exp, axis=1)[:, np.newaxis]

    return predictions_exp/predictions_sum


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''

    rows_count = target_index.shape[0]

    return np.sum(-np.log(probs[range(rows_count), target_index]))/rows_count


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''

    rows_count = target_index.shape[0]
    probas = softmax(predictions)
    loss = cross_entropy_loss(probas, target_index)
    
    probas[range(rows_count), target_index] -= 1
    probas /= rows_count

    return loss, probas


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    loss = reg_strength + np.sum(W**2, axis=1)

    return np.sum(loss)/W.shape[0], 2*W/W.shape[0]
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = X@W

    loss, dSoftmax = softmax_with_cross_entropy(predictions, target_index)
    
    return loss, X.T@dSoftmax


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5, epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            batch_indices = batches_indices[np.random.randint(0, len(batches_indices))]
            batch = X[batch_indices, :]
            batch_target = y[batch_indices]
            ce_loss, d_ce_loss = linear_softmax(batch, self.W, batch_target)
            reg_loss, dreg_loss = l2_regularization(self.W, reg)
            loss = ce_loss + reg_loss
            dW = d_ce_loss + dreg_loss
            self.W -= dW*learning_rate

            loss_history.append(loss)
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        return np.argmax(X@self.W, axis=1)
