from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import mglearn

# k-means算法
def simpleKMeans():
    x, y = make_blobs(random_state=1)

    # 簇中心设置为 3 个
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(x)
    # 进行分类 kmeans.predict(x)
    mglearn.discrete_scatter(x[:, 0], x[:, 1], kmeans.labels_, markers='o')
    mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2], markers='^', markeredgewidth=2)


if __name__ == '__main__':
    simpleKMeans()