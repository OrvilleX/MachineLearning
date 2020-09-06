from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import mglearn


# MLP多层感知机
def defaultMlp():
    x, y = make_moons(n_samples=100, noise=0.25, random_state=3)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state= 42)

    # mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])

    # 包含两层各10单元的隐藏层
    # mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 10])
    # 使用tanh的两层10单元的隐藏层
    mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 10],activation='tanh')
    mlp.fit(x_train, y_train)
    mglearn.plots.plot_2d_separator(mlp, x_train, fill=True, alpha=.3)
    mglearn.discrete_scatter(x_train[:, 0], x_train[:, 1], y_train)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')

    plt.show()


if __name__ == '__main__':
    defaultMlp()