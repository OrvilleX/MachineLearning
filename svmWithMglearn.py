from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import mglearn


def showSvm():
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


def showSvmWithPre():
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
    showSvmWithPre()
    #showSvm()