from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_moons, load_breast_cancer
import matplotlib.pyplot as plt
import mglearn


# 随机森林算法
def randomForest():
    x, y = make_moons(n_samples=100, noise=0.25, random_state=3)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)
    forest = RandomForestClassifier(n_estimators=5, random_state=2)
    forest.fit(x_train, y_train)

    fig, axes = plt.subplots(2, 3, figsize=(20,10))
    for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
        ax.set_title("Tree {}".format(i))
        mglearn.plots.plot_tree_partition(x_train, y_train, tree, ax=ax)
    mglearn.plots.plot_2d_separator(forest, x_train, fill=True, ax=axes[-1, -1], alpha=.4)
    axes[-1, -1].set_title("Random Forest")
    mglearn.discrete_scatter(x_train[:,0], x_train[:, 1], y_train)


# 梯度提升树
def gradient():
    cancer = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    gbrt = GradientBoostingClassifier(random_state=0)
    gbrt.fit(x_train, y_train)

    print("Accuracy on training set: {:.3f}".format(gbrt.score(x_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(gbrt.score(x_test, y_test)))

    gbrt1 = GradientBoostingClassifier(random_state=0, max_depth=1)
    gbrt1.fit(x_train, y_train)

    print("Accuracy1 on training set: {:.3f}".format(gbrt1.score(x_train, y_train)))
    print("Accuracy1 on test set: {:.3f}".format(gbrt1.score(x_test, y_test)))

    gbrt2 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
    gbrt2.fit(x_train, y_train)

    print("Accuracy2 on training set: {:.3f}".format(gbrt2.score(x_train, y_train)))
    print("Accuracy2 on test set: {:.3f}".format(gbrt2.score(x_test, y_test)))


if __name__ == '__main__':
    #randomForest()
    gradient()