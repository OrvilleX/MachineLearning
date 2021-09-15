from sklearn.linear_model import LogisticRegression
from numpy import *
from utils import tool


def sigmoid(intx: int):
    """
    S函数
    """
    if intx >= 0:
        return 1.0/(1+exp(-intx))
    else:
        return exp(intx) / (1 + exp(intx))


def gradascent(datamatin, classlabels):
    """
    梯度提升算法
    """
    datamatrix = mat(datamatin)
    labelmat = mat(classlabels).transpose()
    m,n = shape(datamatrix)
    alpha = 0.001
    maxcycles = 500
    weights = ones((n, 1))
    for k in range(maxcycles):
        h = sigmoid(datamatrix * weights)
        error = (labelmat - h)
        weights = weights + alpha * datamatrix.transpose() * error
    return weights


def stocgradascent(datamatrix, classlabels, numiter = 150):
    """
    随机梯度提升算法
    """
    m,n = shape(datamatrix)
    weights = ones(n)
    for j in range(numiter):
        dataindex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randindex = int(random.uniform(0, len(dataindex)))
            tol = sum(datamatrix[randindex] * weights)
            h = sigmoid(tol)
            error = classlabels[randindex] - h
            weights = weights + alpha * error * datamatrix[randindex]
            del(dataindex[randindex])
    return weights


def classifyvector(inx, weights):
    """
    计算结果
    """
    prob = sigmoid(sum(inx*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colictest():
    trainingset, traininglabels = tool.file2floatMatrix('horseColicTraining.txt', 21)
    testset, testlabels = tool.file2floatMatrix('horseColicTest.txt', 21)
    trainweights = stocgradascent(trainingset, traininglabels, 500)
    errorcount = 0
    numtestvec = 0.0
    for i in range(testset.shape[0]):
        numtestvec += 1
        if int(classifyvector(testset[i], trainweights)) != testlabels[i]:
            errorcount += 1
    errorrate = (float(errorcount) / numtestvec)
    print("the error rate of this test is: %f" % errorrate)
    return errorrate


def testLogistic():
    """
    基于sklearn的Logistic回归
    """
    logreg = LogisticRegression(C=1)
    trainingset, traininglabels = tool.file2floatMatrix('horseColicTraining.txt', 21)
    testset, testlabels = tool.file2floatMatrix('horseColicTest.txt', 21)
    logreg.fit(trainingset, traininglabels)
    print("logreg.coef_: {}".format(logreg.coef_))
    print("logreg.intercept_: {}".format(logreg.intercept_))
    print("Training set score: {:.2f}".format(logreg.score(trainingset, traininglabels)))
    print("Test set score: {:.2f}".format(logreg.score(testset, testlabels)))


if __name__ == '__main__':
    colictest()
    # testLogistic()