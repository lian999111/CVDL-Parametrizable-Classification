# %%
import tensorflow as tf 
import tensorflow.keras as k 
import numpy as np 
import matplotlib.pyplot as plt
import imageio
import glob
import utils


# %% Load model
model = k.models.load_model('log/tripletloss/TL_model_v4_dim_64_batch_32_num_6_lr_0005_all.h5')


# %% Load images
test_imgs = []
for im_path in glob.glob("test_imgs/*.jpg"):
     img = -imageio.imread(im_path)
     plt.imshow(img, cmap='gray')
     plt.show()
     test_imgs.append(img)

test_imgs = np.array(test_imgs)
test_imgs = np.expand_dims(test_imgs, axis=-1)
test_imgs = test_imgs / 255.0
# test_imgs = 1 - test_imgs     # invert the image

# %% Compare test images
encodings = model.predict(test_imgs)
pairwise_dists = utils.cal_pairwise_dists(encodings)
classification = pairwise_dists < 2.8


# %%