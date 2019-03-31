def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    
    true_predictions = prediction[prediction == ground_truth]
    true_positives = true_predictions[true_predictions == True]
    right_answers = ground_truth[ground_truth == True]
    all_positives = prediction[prediction == True]
    precision = len(true_positives)/len(all_positives)
    recall = len(true_positives)/len(right_answers)
    accuracy = len(true_predictions)/len(prediction)
    f1 = 2*precision*recall/(precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    true_predictions = prediction[prediction == ground_truth]
    return len(true_predictions)/len(prediction)
