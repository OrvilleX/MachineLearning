from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import mglearn


# 聚类算法对比评估
def ariAndnmiTest(score):
    """
    ARI与NMI
    """
    x, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    scaler = StandardScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks':(), 'yticks': ()})
    algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]

    random_state = np.random.RandomState(seed=0)
    random_clusters = random_state.randint(low=0, high=2, size=len(x))

    axes[0].scatter(x_scaled[:, 0], x_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60)
    axes[0].set_title("Random assignment - ARI: {:.2f}".format(score(y, random_clusters)))

    for ax, algorithm in zip(axes[1:], algorithms):
        clusters = algorithm.fit_predict(x_scaled)
        ax.scatter(x_scaled[:, 0], x_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60)
        ax.set_title("{} - ARI: {:.2f}".format(algorithm.__class__.__name__, score(y, clusters)))
    plt.show()


def silTest():
    """
    轮廓系数
    """
    x, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    scaler = StandardScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks':(), 'yticks': ()})
    algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]

    random_state = np.random.RandomState(seed=0)
    random_clusters = random_state.randint(low=0, high=2, size=len(x))

    axes[0].scatter(x_scaled[:, 0], x_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60)
    axes[0].set_title("Random assignment - ARI: {:.2f}".format(silhouette_score(x_scaled, random_clusters)))

    for ax, algorithm in zip(axes[1:], algorithms):
        clusters = algorithm.fit_predict(x_scaled)
        ax.scatter(x_scaled[:, 0], x_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60)
        ax.set_title("{} - ARI: {:.2f}".format(algorithm.__class__.__name__, silhouette_score(x_scaled, clusters)))
    plt.show()


if __name__ == '__main__':
    # ariAndnmiTest(adjusted_rand_score) # ARI
    # ariAndnmiTest(normalized_mutual_info_score) # NMI
    silTest()
