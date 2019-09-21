
# coding: utf-8

# In[5]:

# username: Ravi Teja Sunkara
# UBIT Name: rsunkara
# UBIT Number: 50292191

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mlxtend.data import loadlocal_mnist


# In[6]:


# Loading MNIST data into arrays
x1, y1 = loadlocal_mnist(images_path = './data/train-images.idx3-ubyte', 
                         labels_path = './data/train-labels-idx1-ubyte')

x2, y2 = loadlocal_mnist(images_path = './data/t10k-images.idx3-ubyte', 
                         labels_path = './data/t10k-labels.idx1-ubyte')


# In[7]:


print('Dimensions: %s x %s' % (x1.shape[0], x1.shape[1]))
print('Dimensions: %s x %s' % (x2.shape[0], x2.shape[1]))

print('Labels: %s' % np.unique(y1))
print('Class Distribution: %s' % np.bincount(y1))


# In[8]:


# Storing as csv files
np.savetxt(fname = './train_images.csv', 
           X=x1, delimiter=',', fmt='%d')
np.savetxt(fname = './train_labels.csv', 
           X=y1, delimiter=',', fmt='%d')

np.savetxt(fname = './test_images.csv', 
           X=x2, delimiter=',', fmt='%d')
np.savetxt(fname = './test_labels.csv', 
           X=y2, delimiter=',', fmt='%d')


# In[10]:


# Creating Data Frames
train_images = pd.read_csv('train_images.csv')
train_labels = pd.read_csv('train_labels.csv')
test_images = pd.read_csv('test_images.csv')
test_labels = pd.read_csv('test_labels.csv')


# In[11]:


train = train_images
train['labels'] = train_labels['5'].astype(int)


# In[12]:


# finding mean 
train_sum = train.groupby('labels').sum()
train_count = train.groupby('labels').count()
train_mean = train_sum/train_count


# ### Mean training images

# In[178]:


train_mean_list = train_mean.values
figure(num=None, figsize=(10, 10))
fig = plt.figure()
fig.add_subplot(251)
plt.imshow(train_mean_list[0].reshape((28,28)), cmap='gray')
fig.add_subplot(252)
plt.imshow(train_mean_list[1].reshape((28,28)), cmap='gray')
fig.add_subplot(253)
plt.imshow(train_mean_list[2].reshape((28,28)), cmap='gray')
fig.add_subplot(254)
plt.imshow(train_mean_list[3].reshape((28,28)), cmap='gray')
fig.add_subplot(255)
plt.imshow(train_mean_list[4].reshape((28,28)), cmap='gray')
fig.add_subplot(256)
plt.imshow(train_mean_list[5].reshape((28,28)), cmap='gray')
fig.add_subplot(257)
plt.imshow(train_mean_list[6].reshape((28,28)), cmap='gray')
fig.add_subplot(258)
plt.imshow(train_mean_list[7].reshape((28,28)), cmap='gray')
fig.add_subplot(259)
plt.imshow(train_mean_list[8].reshape((28,28)), cmap='gray')
fig.add_subplot(2,5,10)
plt.imshow(train_mean_list[9].reshape((28,28)), cmap='gray')


# ## Standard Deviation

# In[14]:


var_dict = {}
for i in np.unique(train['labels']).tolist():
    var_dict[i] = train[train.labels==i]
    
lab_dict = {}
for i in var_dict:
    lab_dict[i] = var_dict[i]['labels']
    del var_dict[i]['labels']


# In[15]:


zero = var_dict[0].reset_index()
one = var_dict[1].reset_index()
two = var_dict[2].reset_index()
three = var_dict[3].reset_index()
four = var_dict[4].reset_index()
five = var_dict[5].reset_index()
six = var_dict[6].reset_index()
seven = var_dict[7].reset_index()
eight = var_dict[8].reset_index()
nine = var_dict[9].reset_index()


# In[16]:


# for images of 0
zero.drop('index', axis=1, inplace=True)
zero.head(5)

for i in range(len(zero)):
    for j in range(len(train_mean.iloc[0])):
        zero.iloc[i, j] = (zero.iloc[i, j] - train_mean.iloc[0, j])**2
        
zero_sum = (pd.DataFrame(zero.sum(axis=0))).transpose()
zero_sumbyN = zero_sum.div(len(zero))
zero_std = np.sqrt(zero_sumbyN)


# In[17]:


# for images of 1
one.drop('index', axis=1, inplace=True)
one.head(5)

for i in range(len(one)):
    for j in range(len(train_mean.iloc[1])):
        one.iloc[i, j] = (one.iloc[i, j] - train_mean.iloc[1, j])**2

one_sum = (pd.DataFrame(one.sum(axis=0))).transpose()
one_sumbyN = one_sum.div(len(one))
one_std = np.sqrt(one_sumbyN)


# In[18]:


# for images of 2
two.drop('index', axis=1, inplace=True)
two.head(5)
for i in range(len(two)):
    for j in range(len(train_mean.iloc[2])):
        two.iloc[i, j] = (two.iloc[i, j] - train_mean.iloc[2, j])**2

two_sum = (pd.DataFrame(two.sum(axis=0))).transpose()
two_sumbyN = two_sum.div(len(two))
two_std = np.sqrt(two_sumbyN)


# In[19]:


# for images of 3
three.drop('index', axis=1, inplace=True)
three.head(5)
for i in range(len(three)):
    for j in range(len(train_mean.iloc[3])):
        three.iloc[i, j] = (three.iloc[i, j] - train_mean.iloc[3, j])**2

three_sum = (pd.DataFrame(three.sum(axis=0))).transpose()
three_sumbyN = three_sum.div(len(three))
three_std = np.sqrt(three_sumbyN)


# In[20]:


# for images of 4
four.drop('index', axis=1, inplace=True)
four.head(5)
for i in range(len(four)):
    for j in range(len(train_mean.iloc[4])):
        four.iloc[i, j] = (four.iloc[i, j] - train_mean.iloc[4, j])**2

four_sum = (pd.DataFrame(four.sum(axis=0))).transpose()
four_sumbyN = four_sum.div(len(four))
four_std = np.sqrt(four_sumbyN)


# In[ ]:


# for images of 5
five.drop('index', axis=1, inplace=True)
five.head(5)
for i in range(len(five)):
    for j in range(len(train_mean.iloc[5])):
        five.iloc[i, j] = (five.iloc[i, j] - train_mean.iloc[5, j])**2

five_sum = (pd.DataFrame(five.sum(axis=0))).transpose()
five_sumbyN = five_sum.div(len(five))
five_std = np.sqrt(five_sumbyN)


# In[ ]:


# for images of 6
six.drop('index', axis=1, inplace=True)
six.head(5)
for i in range(len(six)):
    for j in range(len(train_mean.iloc[6])):
        six.iloc[i, j] = (six.iloc[i, j] - train_mean.iloc[6, j])**2

six_sum = (pd.DataFrame(six.sum(axis=0))).transpose()
six_sumbyN = six_sum.div(len(six))
six_std = np.sqrt(six_sumbyN)


# In[ ]:


# for images of 7
seven.drop('index', axis=1, inplace=True)
seven.head(5)
for i in range(len(seven)):
    for j in range(len(train_mean.iloc[7])):
        seven.iloc[i, j] = (seven.iloc[i, j] - train_mean.iloc[7, j])**2

seven_sum = (pd.DataFrame(seven.sum(axis=0))).transpose()
seven_sumbyN = seven_sum.div(len(seven))
seven_std = np.sqrt(seven_sumbyN)


# In[ ]:


# for images of 8
eight.drop('index', axis=1, inplace=True)
eight.head(5)
for i in range(len(eight)):
    for j in range(len(train_mean.iloc[8])):
        eight.iloc[i, j] = (eight.iloc[i, j] - train_mean.iloc[8, j])**2

eight_sum = (pd.DataFrame(eight.sum(axis=0))).transpose()
eight_sumbyN = eight_sum.div(len(eight))
eight_std = np.sqrt(eight_sumbyN)


# In[ ]:


# for images of 9
nine.drop('index', axis=1, inplace=True)
nine.head(5)
for i in range(len(nine)):
    for j in range(len(train_mean.iloc[9])):
        nine.iloc[i, j] = (nine.iloc[i, j] - train_mean.iloc[9, j])**2

nine_sum = (pd.DataFrame(nine.sum(axis=0))).transpose()
nine_sumbyN = nine_sum.div(len(nine))
nine_std = np.sqrt(nine_sumbyN)


# ### Standard Deviation Images

# In[ ]:


stdev = pd.concat([zero_std, one_std, two_std, three_std, four_std, five_std, six_std, seven_std, eight_std, nine_std], axis=0)

stdev_list = stdev.values
figure(num=None, figsize=(10, 10))
fig = plt.figure()
fig.add_subplot(251)
plt.imshow(stdev_list[0].reshape((28,28)), cmap='gray')
fig.add_subplot(252)
plt.imshow(stdev_list[1].reshape((28,28)), cmap='gray')
fig.add_subplot(253)
plt.imshow(stdev_list[2].reshape((28,28)), cmap='gray')
fig.add_subplot(254)
plt.imshow(stdev_list[3].reshape((28,28)), cmap='gray')
fig.add_subplot(255)
plt.imshow(stdev_list[4].reshape((28,28)), cmap='gray')
fig.add_subplot(256)
plt.imshow(stdev_list[5].reshape((28,28)), cmap='gray')
fig.add_subplot(257)
plt.imshow(stdev_list[6].reshape((28,28)), cmap='gray')
fig.add_subplot(258)
plt.imshow(stdev_list[7].reshape((28,28)), cmap='gray')
fig.add_subplot(259)
plt.imshow(stdev_list[8].reshape((28,28)), cmap='gray')
fig.add_subplot(2,5,10)
plt.imshow(stdev_list[9].reshape((28,28)), cmap='gray')


# ## LDA on MNIST    
# Referred from: https://github.com/solopku/MNIST/blob/master/lda.py

# In[ ]:


get_ipython().run_line_magic('load', 'pca')


# In[3]:


'''
Cai Shaofeng - 2017.3
Implementation of Linear Discriminant Analysis
'''

from data_loader import *
from plot_utilities import *
from numpy.linalg import eigh, pinv
from sklearn.neighbors import KNeighborsClassifier
# from pca import *

class LDA(object):
    def __init__(self):
        pass

    def fit(self, trainX, trainY):
        self.class_sign, self.class_count = np.unique(trainY, return_counts=True)
        self.m, self.n, self.class_num = trainX.shape[0], trainX.shape[1], self.class_sign.shape[0]
        # mean_vector/mean_vectors of the data
        self.mean_vec = np.mean(trainX, axis=0).reshape(1, self.n)
        self.mean_vecs = np.ndarray(shape=(self.class_num, self.n))
        for cls_idx in range(self.class_num):
            self.mean_vecs[cls_idx] = np.mean(trainX[trainY==cls_idx], axis=0)

        # calc the Sw and Sb
        Sb, Sw = np.zeros(shape=(self.n, self.n), dtype='float32'), np.zeros(shape=(self.n, self.n), dtype='float32')
        for cls_idx in range(self.class_num):
            # update Sb
            centered_mean = (self.mean_vecs[cls_idx] - self.mean_vec).reshape((self.n, 1))
            Sb += self.class_count[cls_idx] * np.dot(centered_mean, centered_mean.T)

            # update Sw
            cls_sampels = trainX[(trainY==cls_idx)]
            Sw += np.dot(cls_sampels.T, cls_sampels)

        # calc S
        S = np.dot(pinv(Sw), Sb)
        # eigen decomposition
        self.eigen_vals, self.eigen_vecs = eigh(S)
        # sort the eigenvalue in descending order, actually eigh already reture the ascending order ones
        idx = np.argsort(self.eigen_vals)[::-1]
        self.eigen_vals, self.eigen_vecs = self.eigen_vals[idx], self.eigen_vecs[:, idx]
        return self

    def dimension_reduction(self, trainX, dimension=2):
        return np.dot(trainX - self.mean_vec, self.eigen_vecs[:, :dimension])


if __name__ == '__main__':
    # load data
    trainX, trainY = load_data('training')
    testX, testY = load_data('testing')
    trainX, testX = trainX/255.0, testX/255.0  # normalize pixel value into [0, 1]

    # fit the LDA, eigen decomposition
    lda = LDA().fit(trainX, trainY)

    # 2d and 3d projection
    reduced_X = lda.dimension_reduction(trainX, 3)
    plot_2d(reduced_X[:, 0], reduced_X[:, 1], label=trainY, savePath='figs/lda_lda_2d_projection')
    plot_3d(reduced_X[:, 0], reduced_X[:, 1], reduced_X[:, 2], label=trainY, savePath='figs/lda_lda_3d_projection')

    # show first 2 and 3 eigenvectors
    show_eigenvecs(lda, 2, 'figs/lda_2_eigenvecs')
    show_eigenvecs(lda, 3, 'figs/lda_3_eigenvecs')

    # show all eigenvectors
    fig = plt.figure(figsize=(10, 10))
    for idx in range(9):
        a = fig.add_subplot(3, 3, idx + 1)
        plt.imshow(lda.eigen_vecs[:, idx].reshape((28, 28)), cmap='Greys')
        plt.axis('off')
    # plt.show()
    plt.savefig('figs/lda_all_eigenvecs.eps', format='eps', dpi=100);plt.clf()

    # reduce dimension to 1-9, and classify with KNN
    lda_accuracy, max_dimension = [], 20
    for dimension in range(1, max_dimension+1):
        reduced_trainX, reduced_testX = lda.dimension_reduction(trainX, dimension), lda.dimension_reduction(testX, dimension)
        KNN = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
        KNN.fit(reduced_trainX, trainY)
        lda_accuracy.append(KNN.score(reduced_testX, testY))
        print('dimension: %4d\t|\taccuracy: %.6f' % (dimension, lda_accuracy[-1]))

    plt.plot(range(len(lda_accuracy)), lda_accuracy, marker='*')
    plt.axvline(x=9, color='r');plt.xticks([0, 5, 9, 10, 15, 20])
    plt.xlabel('dimensionality');plt.ylabel('accuracy');plt.title('accuracy VS. dimensionality');
    plt.savefig('figs/lda_acc_dimension.eps', format='eps', dpi=100)

