from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import mglearn
from sklearn.preprocessing import StandardScaler


def simpleTest():
    x, y =  make_moons(n_samples=200, noise=0.05, random_state=0)
    scaler = StandardScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)

    dbscan = DBSCAN() # 支持eps与min_samples 参数设置
    clusters = dbscan.fit_predict(x_scaled)
    plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()

if __name__ == '__main__':
    simpleTest()