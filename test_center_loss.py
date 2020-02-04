# %%
import tensorflow as tf 
import tensorflow.keras as k 
import numpy as np 
import matplotlib.pyplot as plt
import imageio
import glob
import utils


# %% Load model
model = k.models.load_model('log/centerloss/model_v4_64.h5')


# %% Load images
test_imgs = []
for im_path in glob.glob("test_imgs/*.jpg"):
     img = 255-imageio.imread(im_path)  # invert the image
     # plt.imshow(img, cmap='gray')
     # plt.show()
     test_imgs.append(img)

test_imgs = np.array(test_imgs, dtype=np.float)
test_imgs = np.expand_dims(test_imgs, axis=-1)
test_imgs = np.minimum(test_imgs*1.5, 255.0)
test_imgs = test_imgs / 255.0


# %% Compare test images
threshold = 0.5
encodings = model.predict(test_imgs)
pairwise_dists = utils.cal_pairwise_dists(encodings)
classification = pairwise_dists < threshold


# %%
import openpyxl
wb = openpyxl.load_workbook(filename='test_imgs/test_img_template.xlsx')
ws = wb.worksheets[0]
for (i, j), value in np.ndenumerate(classification):
     ws.cell(row=i+2, column=j+2).value = value
wb.save('log/centerloss/accuracy_model_v4_64.xlsx')

# %%
