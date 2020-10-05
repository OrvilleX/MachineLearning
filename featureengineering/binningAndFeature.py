import pandas as pd
import matplotlib.pyplot as plt
import mglearn.datasets
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures

def binningMain():
    """
    基于pandas的分箱
    """
    x, y = mglearn.datasets.make_wave(n_samples=100)
    # 制作箱子
    bins = np.linspace(-3, 3, 11)
    print("bins: {}".format(bins))

    # 分箱
    which_bin = np.digitize(x, bins=bins)
    print("\nData points:\n", x[:5])
    print("\nBin membership for data ppints:\n", which_bin[:5])

    # 对分箱后的数据进行one-hot
    encoder = OneHotEncoder()
    encoder.fit(which_bin)
    x_binned = encoder.transform(which_bin)
    print(x_binned[:5])

    # 进行模型训练测试
    line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
    line_binned = encoder.transform(np.digitize(line, bins=bins))
    reg = LinearRegression().fit(x_binned, y)
    plt.plot(line, reg.predict(line_binned), label='linear regression binned')

    reg = DecisionTreeRegressor(min_samples_split=3).fit(x_binned, y)
    plt.plot(line, reg.predict(line_binned), label='decision tree binned')
    plt.plot(x[:, 0], y, 'o', c='k')
    plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
    plt.legend(loc="best")
    plt.ylabel('Regression output')
    plt.xlabel('Input feature')
    plt.show()


def interactionMain():
    """
    交互特征
    """
    x, y = mglearn.datasets.make_wave(n_samples=100)
    bins = np.linspace(-3, 3, 11)
    which_bin = np.digitize(x, bins=bins)
    encoder = OneHotEncoder()
    encoder.fit(which_bin)
    x_binned = encoder.transform(which_bin)
    line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
    line_binned = encoder.transform(np.digitize(line, bins=bins))

    # 交互特征1
    x_combined = np.hstack([x, x_binned.A])
    # 交互特征2
    x_combined = np.hstack([x_binned.A, x * x_binned.A])

    # 进行算法建模与测试
    reg = LinearRegression().fit(x_combined, y)

    # 交互特征1
    # line_combined = np.hstack([line, line_binned.A])
    # 交互特征2
    line_combined = np.hstack([line_binned.A, line * line_binned.A])
    plt.plot(line, reg.predict(line_combined), label='linear regression combined')

    for bin in bins:
        plt.plot([bin, bin], [-3, 3], ':', c='k')

    plt.legend(loc="best")
    plt.ylabel("Regression output")
    plt.xlabel("Input feature")
    plt.plot(x[:, 0], y, 'o', c='k')
    plt.show()


def polynomialMain():
    """
    多项式特征
    """
    x, y = mglearn.datasets.make_wave(n_samples=100)
    line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
    poly = PolynomialFeatures(degree=10, include_bias=False)
    poly.fit(x)
    x_poly = poly.transform(x)
    reg = LinearRegression().fit(x_poly, y)
    line_poly = poly.transform(line)
    plt.plot(line, reg.predict(line_poly), label='polynomial linear regression')
    plt.plot(x[:, 0], y, 'o', c='k')
    plt.ylabel("Regression output")
    plt.xlabel("Input Feature")
    plt.legend(loc="best")
    plt.show()



if __name__ == '__main__':
    polynomialMain()
    # interactionMain()
    # binningMain()