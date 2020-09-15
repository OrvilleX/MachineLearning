from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import mglearn


def simpleTest():
    x, y = make_blobs(random_state=1)
    agg = AgglomerativeClustering(n_clusters=3)
    assignment = agg.fit_predict(x)

    mglearn.discrete_scatter(x[:, 0], x[:, 1], assignment)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.show()


if __name__ == '__main__':
    simpleTest()