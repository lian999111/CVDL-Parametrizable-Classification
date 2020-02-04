# CVDL-Parametrizable-Classification
The aim of this project is to study parameterizble classifier using the concept of Siamese Networks.
Performance of different losses are tested, including triplet loss and center loss.

# How to use
This project exploits 2 methods, triplet loss and center loss, to train a siamese network that learns to identify the similarity between 2 images. The project is written using **Tensorflow v2.0**. We recommend to run the codes in **Interactive Window** in Visual Code as we write the them in a cell-by-cell fashion.

The repository is organized in such a way:
1. The main training, including data pre-processing, and analysis of both methods are done in **main_triplet_loss.py** and **main_center_loss.py**

2. The last cell of each above-mentioned script is the saving of .tsv files of encoding vectors and metadata for tensorboard embedding projector. To show them in tensorboard, use the command in the command window:

+ tensorboard --logdir PATH_TO_FILES (e.g. tensorboard --logdir log/centerloss)

+ Note: Make sure the directory contains **projector_config.pbtxt**, **mnist_sprite.jpg**, **feature_vecs.tsv**, and **metadata.tsv**

3. The testing on images created by us are done in **test_triplet_loss.py** and **test_center_loss.py**. The testing results are stored as .xlsx where 1 indicates positive classification.