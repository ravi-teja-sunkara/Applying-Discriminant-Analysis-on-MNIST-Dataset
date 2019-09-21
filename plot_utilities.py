# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:29:33 2019

@author: sravi
"""

'''
Cai Shaofeng - 2017.3
Implementation of plot utilities
'''

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import matplotlib
import numpy as np

def show_img(img):
    plt.imshow(img, cmap='Greys')
    plt.show()

def show_eigenvecs(pca, num, savePath=None):
    fig = plt.figure(figsize=(10, 10))
    for idx in range(num):
        a = fig.add_subplot(1, num, idx+1)
        plt.imshow(pca.eigen_vecs[:, idx].reshape((28, 28)), cmap='Greys')
        plt.axis('off')
    if savePath == None:
        plt.show()
    else:
        plt.savefig(savePath + '.eps', format='eps', dpi=1000)
    plt.clf()

def plot_2d(X, Y, label, title='Projected Data in 2d Plot', savePath=None):
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 10))
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(X, Y, c=label, cmap=matplotlib.colors.ListedColormap(colors), edgecolors='none')
    cb = plt.colorbar()
    loc = np.arange(0, max(label), max(label) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(np.arange(0, 10))
    plt.title(title)
    plt.xticks([]);plt.yticks([])
    if savePath == None:
        plt.show()
    else:
        plt.savefig(savePath + '.eps', format='eps', dpi=1000)

def plot_3d(X, Y, Z, label, title='Projected Data in 3d Plot', savePath=None):
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 10))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    bar = ax.scatter(X, Y, Z, c=label, cmap=matplotlib.colors.ListedColormap(colors), edgecolors='none')

    cb = plt.colorbar(bar)
    loc = np.arange(0, max(label), max(label) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(np.arange(0, 10))
    plt.title(title)
    plt.xticks([]);plt.yticks([]);ax.set_zticks([])
    if savePath == None:
        plt.show()
    else:
        plt.savefig(savePath + '.eps', format='eps', dpi=1000)
