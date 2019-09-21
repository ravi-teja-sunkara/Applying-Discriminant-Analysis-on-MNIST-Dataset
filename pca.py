'''
Cai Shaofeng - 2017.3
Implementation of Principle Component Analysis
'''

from data_loader import *
from plot_utilities import *
from numpy.linalg import eigh
from sklearn.neighbors import KNeighborsClassifier

class PCA(object):
    def __init__(self):
        pass
    
    def fit(self, trainX):
        self.m, self.n = trainX.shape
        # mean vector of the data
        self.mean_vec = np.mean(trainX, axis=0).reshape(1, self.n)
        # center the data, and calc the covariance matrix
        S = np.cov(trainX-self.mean_vec, rowvar=False)
        # eigen decomposition
        self.eigen_vals, self.eigen_vecs = eigh(S)
        # sort the eigenvalue in descending order, actually eigh already reture the ascending order ones
        idx = np.argsort(self.eigen_vals)[::-1]
        self.eigen_vals, self.eigen_vecs = self.eigen_vals[idx], self.eigen_vecs[:, idx]
        return self

    def dimension_reduction(self, trainX, dimension=2):
        return np.dot(trainX-self.mean_vec, self.eigen_vecs[:, :dimension])


if __name__ == '__main__':
    # load data
    trainX, trainY = load_data('training')
    testX, testY = load_data('testing')
    trainX, testX = trainX/255.0, testX/255.0 # normalize pixel value into [0, 1]

    pca = PCA()
    pca.fit(trainX)

    # 2d and 3d projection
    reduced_X = pca.dimension_reduction(trainX, 3)
    plot_2d(reduced_X[:, 0], reduced_X[:, 1], label=trainY, savePath='figs/pca_2d_projection')
    plot_3d(reduced_X[:, 0], reduced_X[:, 1], reduced_X[:, 2], label=trainY, savePath='figs/pca_3d_projection')

    # show first 2 and 3 eigenvectors
    show_eigenvecs(pca, 2, 'figs/pca_2_eigenvecs')
    show_eigenvecs(pca, 3, 'figs/pca_3_eigenvecs')

    # show all eigenvectors
    fig = plt.figure(figsize=(10, 10))
    for idx in range(784):
        a = fig.add_subplot(28, 28, idx+1)
        plt.imshow(pca.eigen_vecs[:, idx].reshape((28, 28)), cmap='Greys')
        plt.axis('off')
    #plt.show()
    plt.savefig('figs/pca_all_eigenvecs.eps', format='eps', dpi=100)

    # reduce dimension to 40, 80 and 200, and classify with KNN
    for dimension in [40, 80, 200]:
        reduced_trainX, reduced_testX = pca.dimension_reduction(trainX, dimension), pca.dimension_reduction(testX, dimension)
        KNN = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
        KNN.fit(reduced_trainX, trainY)
        print 'dimension: %4d\t|\taccuracy: %.6f' %(dimension, KNN.score(reduced_testX, testY))

    # find the d to preserve over 95% of total energy, then apply to KNN\
    d, acc_energy, pca.eigen_vals = 0, [0.0], pca.eigen_vals/np.sum(pca.eigen_vals)
    while acc_energy[-1] < 0.95:
        acc_energy.append(pca.eigen_vals[d]+acc_energy[-1])
        d += 1

    for index in xrange(len(acc_energy)):
        print index, acc_energy[index]

    plt.plot(range(len(acc_energy)), acc_energy, 'b', label='accumulate lamdba')
    plt.plot(range(len(acc_energy)), pca.eigen_vals[:len(acc_energy)], 'r', label='lamdba')
    plt.xlabel('index');
    plt.ylabel('lambda');
    plt.title('lambda VS preserved energy');
    plt.legend(loc='upper left')
    # plt.show()
    plt.savefig('figs/labmda_vs_energy.eps', format='eps', dpi=100)

    print 'need %d dimensions to preserve over 95%%(%.6f) of total energy, processing...' %(d, acc_energy[-1])
    reduced_trainX, reduced_testX = pca.dimension_reduction(trainX, d), pca.dimension_reduction(testX, d)
    KNN = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    KNN.fit(reduced_trainX, trainY)
    print 'dimension: %4d\t|\taccuracy: %.6f' %(d, KNN.score(reduced_testX, testY))