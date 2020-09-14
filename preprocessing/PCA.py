from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mglearn

"""
PCA主成分分析
"""

def simplePCA():
    cancer = load_breast_cancer()
    scaler = StandardScaler()
    scaler.fit(cancer.data)
    x_scaled = scaler.transform(cancer.data)
    pca = PCA(n_components=2)
    pca.fit(x_scaled)

    x_pca = pca.transform(x_scaled)
    print("Original shape: {}".format(str(x_scaled.shape)))
    print("Reduced shape: {}".format(str(x_pca.shape)))

    # 二维散点图可视化
    # plt.figure(figsize=(8, 8))
    # mglearn.discrete_scatter(x_pca[:, 0], x_pca[:, 1], cancer.target)
    # plt.legend(cancer.target_names, loc = "best")
    # plt.gca().set_aspect("equal")
    # plt.xlabel("First principal component")
    # plt.ylabel("Second principal component")
    # plt.show()

    # 热图
    plt.matshow(pca.components_, cmap='viridis')
    plt.yticks([0,1], ["First component", "Second component"])
    plt.colorbar()
    plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
    plt.xlabel('Feature')
    plt.ylabel('Principal components')
    plt.show()


if __name__ == '__main__':
    simplePCA()