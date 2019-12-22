import numpy as np
import math
from copy import copy
import matplotlib.pyplot as plt
from DLCVDatasets import get_dataset

def l2_norm_difference(img1, img2):
    img1 = img1.flatten()
    img2 = img2.flatten()
    
    return np.linalg.norm(img1 - img2)

def offline_triplets_selection(training_size, test_size, number_of_batches, used_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    
    x_train, y_train, x_test, y_test, class_names = get_dataset( 'mnist', used_labels, training_size, test_size )
    
    triplets = np.zeros((training_size, 3, 28, 28))
    triplets_idx = 0
    
    alpha = 0
    
    # loop through all classes
    for positive_label in used_labels:
        positive_set = copy(x_train[( y_train==positive_label,)])
        negative_set = copy(x_train[( y_train!=positive_label,)])
        
        elements_in_p_set = len(positive_set)
        elements_in_n_set = len(negative_set)
        
        p_batch_size = math.floor(elements_in_p_set/number_of_batches)
        n_batch_size = math.floor(elements_in_n_set/number_of_batches)
        
        # loop through all batches
        for i,j in zip( range(0, elements_in_p_set, p_batch_size),range(0, elements_in_n_set, n_batch_size) ):
            # prevent going out of indexes
            if i+p_batch_size-1 >= elements_in_p_set:
                i = elements_in_p_set - p_batch_size
            if j+n_batch_size-1 >= elements_in_n_set:
                j = elements_in_n_set - n_batch_size
            
            anchor = positive_set[i,]
            
            # loop inside each positive batch
            for p_idx in range(i+1,i+p_batch_size):
                positive = positive_set[p_idx,]
                
                # get difference between anchor and positive
                diff_p = l2_norm_difference(anchor, positive)
                
                # loop inside negative batch to find hardest valid negative image
                min_diff_n = diff_p + 1000
                semihard_neg_idx = j
                
                for n_idx in range(j,j+n_batch_size):
                    negative = negative_set[n_idx,]
                    
                    # get difference between anchor and proposed negative
                    diff_n = l2_norm_difference(anchor, negative)
                    
                    if (diff_n > (diff_p + alpha)) & (diff_n < min_diff_n):
                        min_diff_n = diff_n
                        semihard_neg_idx = n_idx
                
                negative = negative_set[semihard_neg_idx,]
                
                triplets[triplets_idx,0,] = anchor
                triplets[triplets_idx,1,] = positive
                triplets[triplets_idx,2,] = negative
                triplets_idx = triplets_idx + 1
    return triplets, triplets_idx
            
used_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
training_size = 2500  # None for all images
test_size = 1000  # None for all images
number_of_batches = 100

[triplets, triplets_idx] = offline_triplets_selection(training_size, test_size, number_of_batches, used_labels)

# show results
num_examples = 20
step = 100

plt.figure(figsize=(math.ceil(3 * 0.7), math.ceil(num_examples * 0.75)))

for idx, i in enumerate(range(0,num_examples*step,step)):
    plt.subplot(num_examples, 3, 3*idx + 1)
    plt.axis("off")
    plt.imshow(triplets[i,0,], cmap="Greys")
    
    plt.subplot(num_examples, 3, 3*idx + 2)
    plt.axis("off")
    plt.imshow(triplets[i,1,], cmap="Greys")
    
    plt.subplot(num_examples, 3, 3*idx + 3)
    plt.axis("off")
    plt.imshow(triplets[i,2,], cmap="Greys")

plt.show()