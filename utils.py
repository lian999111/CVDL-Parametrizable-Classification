import numpy as np 
import matplotlib.pyplot as plt

def cal_pairwise_dists(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: numpy array of shape (num_samples, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_dist: numpy array of shape (num_samples, num_samples)
    """

    # Get the dot product between all embeddings
    # shape (num_samples, num_samples)
    dot_product = np.matmul(embeddings, embeddings.T)

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (num_samples,)
    squared_norm = np.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (num_samples, num_samples)
    distances = np.expand_dims(squared_norm, 1) - 2.0 * dot_product + np.expand_dims(squared_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = np.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = np.equal(distances, 0.0).astype(np.float64)
        distances = distances + mask * 1e-16

        distances = np.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

def l2_normalize(embeddings, axis=1):
    """L2 normalize embeddings.

    Args:
        embeddings: Numpy array of shape (num_samples, embed_dim)
        axis: Axis along the embedding dimension

    Returns:
        normalized_embeddings: numpy array of shape (num_samples, embed_dim)
    """
    norm = np.linalg.norm(embeddings, axis=axis)
    mask = np.equal(norm, 0.0).astype(np.float64)
    norm = norm + mask * 1e-16      # avoid division-by-zero
    return embeddings / norm[:, np.newaxis]  # add newaxis for correct broadcasting


def performance_test(pairwise_dists, labels, threshold):
    """Calculate measures of performance.

    Args:
        pairwise_dists: square matrix with distances between embeddings
        labels: labels or the embeddings
        threshold: value below which two embeddings will be taken as equal

    Returns:
        recall: correctly positive classified pairs over all positive pairs
        FAR: incorrectly positive classified pairs over all negative pairs
        precision: correctly positive classified pairs over all positive classified pairs
    """

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    really_equal = np.equal(np.expand_dims(labels, 0), np.expand_dims(labels, 1))
    # Check model prediction using given threshold
    predicted_equal = pairwise_dists < threshold

    number_positive_pairs = np.sum(really_equal)-len(labels)
    number_negative_pairs = pairwise_dists.size - number_positive_pairs - len(labels)

    # Calculate values of confusion matrix
    TP = np.sum(predicted_equal[really_equal]) - len(labels)
    FN = number_positive_pairs - TP
    FP = np.sum(predicted_equal[~really_equal])
    TN = number_negative_pairs - FP

    # Calculate measures of performance
    recall = TP / number_positive_pairs
    FAR = FP / number_negative_pairs
    precision = TP / (TP+FP)

    return recall, FAR, precision

def threshold_evaluation(pairwise_dists, labels, thr_start, thr_end, thr_quantity, i_want_to_plot = True):
    """Calculate measures of performance.

    Args:
        pairwise_dists: square matrix with distances between embeddings
        labels: labels or the embeddings
        thr_start: first threshold to try
        thr_end: last threshold to try
        thr_quantity: number of thresholds to try
        i_want_to_plot: set to true to plot ROC and precision-recall curve

    Returns:
        recall: array of correctly positive classified pairs over all positive pairs per threshold
        FAR: array of incorrectly positive classified pairs over all negative pairs per threshold
        precision: array of correctly positive classified pairs over all positive classified pairs per threshold
    """
    
    # Define thresholds to be used
    threshold_range = np.linspace(thr_start, thr_end, num=thr_quantity)

    # Initialize variables
    recall = np.zeros(len(threshold_range))
    FAR = np.zeros(len(threshold_range))
    precision = np.zeros(len(threshold_range))

    # Get measures of performance and save them in array
    for i, threshold in enumerate(threshold_range):
        recall[i], FAR[i], precision[i] = performance_test(pairwise_dists, labels, threshold)

    # plot measures of performance
    if(i_want_to_plot):
        plt.figure()

        # ROC curve
        plt.subplot(121)
        plt.plot(FAR, recall,'ro',FAR, recall,'b')
        plt.xlabel('FAR')
        plt.ylabel('recall')
        plt.title('ROC curve')
        plt.grid(True)

        # write value of threshold over each point
        for i,(x,y) in enumerate(zip(FAR, recall)):

            label = "{:.2f}".format(threshold_range[i])

            plt.annotate(label, # this is the text
                        (x,y), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext=(0,10), # distance from text to points (x,y)
                        ha='center') # horizontal alignment can be left, right or center
            

        # Precision-recall curve
        plt.subplot(122)
        plt.plot(recall, precision,'ro', recall, precision,'b')
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)

        # write value of threshold over each point
        for i,(x,y) in enumerate(zip(recall, precision)):

            label = "{:.2f}".format(threshold_range[i])

            plt.annotate(label, # this is the text
                        (x,y), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext=(0,10), # distance from text to points (x,y)
                        ha='center') # horizontal alignment can be left, right or center
        
        plt.show()
    return recall, FAR, precision

def get_accuracy_table(pairwise_dists, labels, threshold ):
    """Calculate measures of performance.

    Args:
        pairwise_dists: square matrix with distances between embeddings
        labels: labels or the embeddings
        threshold: value below which two embeddings will be taken as equal

    Returns:
        accuracy_table: table with accuracy (correctly classified over total cases) per class
    """

    accuracy_table = np.zeros((10, 10))
    start_row = 0
    start_column_next = 0

    for row_class in range(10):
        row_span = len(np.argwhere(labels == row_class))
        start_column = start_column_next
        start_column_next = start_column_next + row_span

        for column_class in range(row_class,10):
            column_span = len(np.argwhere(labels == column_class))

            if(row_class == column_class):
                expected_value = 1
            else:
                expected_value = 0
            
            predicted_equal = pairwise_dists[start_row:start_row+row_span-1 , start_column:start_column+column_span-1] < threshold
            
            if(row_class == column_class):
                accuracy = (np.sum(predicted_equal==expected_value)-row_span) / (predicted_equal.size-row_span)
            else:
                accuracy = (np.sum(predicted_equal==expected_value)) / predicted_equal.size
            
            accuracy_table[row_class,column_class] = accuracy
            start_column = start_column + column_span
        
        start_row = start_row + row_span
    
    return accuracy_table