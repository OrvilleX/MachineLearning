from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from numpy import *
from utils import tool


def testLinear():
    """
    基于sklearn的线性支持向量机
    """
    svc = LinearSVC(C=50)
    trainingset, traininglabels = tool.file2floatMatrix('../data/horseColicTraining.txt', 21)
    testset, testlabels = tool.file2floatMatrix('../data/horseColicTest.txt', 21)
    svc.fit(trainingset, traininglabels)
    print("svc.coef_: {}".format(svc.coef_))
    print("svc.intercept_: {}".format(svc.intercept_))
    print("Training set score: {:.2f}".format(svc.score(trainingset, traininglabels)))
    print("Test set score: {:.2f}".format(svc.score(testset, testlabels)))


def showSvm():
    """
    基于核函数的支持向量机
    """
    cancer = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    svc = SVC()
    svc.fit(x_train, y_train)

    print("Accuracy on training set: {:.2f}".format(svc.score(x_train, y_train)))
    print("Accuracy on test set: {:.2f}".format(svc.score(x_test, y_test)))

    plt.plot(x_train.min(axis=0), 'o', label="min")
    plt.plot(x_train.max(axis=0), '^', label="max")
    plt.legend(loc=4)
    plt.xlabel("Feature index")
    plt.ylabel("Feature magnitude")
    plt.yscale("log")
    plt.show();

"""
gamma参数是用于控制高斯核的宽度，它决定了点与点之间“靠近”是指多大的距离
C参数是正则化参数，它限制每个点的重要性
"""


def showSvmWithPre():
    """
    带调参的支持向量机
    """
    cancer = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    min_on_training = x_train.min(axis=0)
    range_on_training = (x_train - min_on_training).max(axis=0)
    x_train_scaled = (x_train - min_on_training) / range_on_training
    x_test_scaled = (x_test - min_on_training) / range_on_training
    svc = SVC(C=1000)
    svc.fit(x_train_scaled, y_train)

    print("Accuracy on training set: {:.3f}".format(svc.score(x_train_scaled, y_train)))
    print("Accuracy on test set: {:.3f}".format(svc.score(x_test_scaled, y_test)))


if __name__ == '__main__':
    testLinear()
    showSvmWithPre()
    showSvm()