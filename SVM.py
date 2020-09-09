from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from numpy import *
import tool


def loaddataset(filename):
    datamat = []
    labelmat = []
    fr = open(filename)
    for line in fr.readlines():
        linearr = line.strip().split('\t')
        datamat.append([float(linearr[0]), float(linearr[1])])
        labelmat.append(float(linearr[2]))
    return datamat, labelmat


class optStruct:
    def __init__(self, datamatin, classlabels, c, toler):
        self.x = datamatin
        self.labelmat = classlabels
        self.c = c
        self.tol = toler
        self.m = shape(datamatin)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.ecache = mat(zeros((self.m, 2)))


def calcek(oS, k):
    fxk = float(multiply(oS.alphas, oS.labelmat).T*(oS.x * oS.x[k, :].T)) + oS.labelmat
    ek = fxk - float(oS.labelmat[k])
    return ek


def selectj(i, oS, ei):
    maxk = -1
    maxdeltae = 0
    ej = 0
    oS.ecache[i] = [1, ei]
    validecachelist = nonzero(oS.ecache[:,0].A)[0]
    if (len(validecachelist)) > 1:
        for k in validecachelist:
            if k == i:
                continue
            ek = calcek(oS, k)
            deltae = abs(ei - ek)
            if (deltae > maxdeltae):
                maxk = k
                maxdeltae = deltae
                ej = ek
        return maxk, ej
    else:
        j = selectjrand(i, oS.m)
        ej = calcek(oS, j)
    return j, ej


def updateek(oS, k):
    ek = calcek(oS, k)
    oS.ecache[k] = [1, ek]


def selectjrand(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0,m))
    return j


def clipalpha(aj, h, l):
    if aj > h:
        aj = h
    if l > aj:
        aj = l
    return aj


def innerl(i, oS):
    ei = calcek(oS, i)
    if ((os.labelMat[i] * ei) < -oS.tol) and (oS.alphas[i] < oS.c) \
        or ((oS.labelmat[i] * ei > oS.tol) and (oS.alphas[i] > 0)):
        j,ej = selectj(i, oS, ei)
        alphaIold = oS.alphas[i].copy()
        alphajold = oS.alphas[j].copy()
        if (oS.labelmat[i] != oS.labelmat[j]):
            l = max(0, oS.alphas[j] - oS.alphas[i])
            h = min(oS.c, oS.c + oS.alphas[j] - oS.alphas[i])
        else:
            l = max(0, oS.alphas[j] + oS.alphas[i] - oS.c)
            h = min(oS.c, oS.alphas[j] + oS.alphas[i])
        if l == h:
            print("l == h")
            return 0
        eta = 2.0 * oS.x[i,:]*oS.x[j, :].T - oS.x[i,:].T - oS.x[j,:]*oS.x[j,:].T
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelmat[j] * (ei - ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], h, l)
        updateek(oS, j)
        if (abs(oS.alphas[j] - alphajold) < 0.00001):
            return 0
        oS.alphas[i] += oS.labelmat[j] - oS.labelmat[i] * (alphajold - oS.alphas[j])
        updateek(oS, i)
        b1 = oS.b - ei - oS.labelmat[i]*(oS.alphas[i] - alphaIold)*oS.x[i,:]*oS.x[j,:].T - oS.labelmat[j]*\
             (oS.alphas[j]-alphajold)*oS.x[j,:]*oS.X[j,:].T
        b2 = oS.b - ej - oS.labelmat[i]*(oS.alphas[i] - alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelmat[j]*\
             (oS.alphas[j]-alphajold)*oS.x[j,:]*oS.x[j,:].T
        if (0 < oS.alphas[j]) and (oS.c > oS.alphas[i]): os.b = b1
        elif (0 < oS.alphas[j]) and (oS.c > oS.alphas[j]): os.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


def testLinear():
    """
    基于sklearn的线性支持向量机
    """
    svc = LinearSVC(C=50)
    trainingset, traininglabels = tool.file2floatMatrix('horseColicTraining.txt', 21)
    testset, testlabels = tool.file2floatMatrix('horseColicTest.txt', 21)
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