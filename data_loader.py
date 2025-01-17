
# coding: utf-8

# In[ ]:


import os
import struct
import numpy as np


# loading the training/testing data
def load_data(dataset = "training", path = "./data"):

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
 #   else:
 #       raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows*cols)

    # get_img = lambda idx: (lbl[idx], img[idx])
    # # Create an iterator which returns each image in turn
    # for i in xrange(len(lbl)):
    #     yield get_img(i)

    return img.astype('float32', copy=True), lbl

