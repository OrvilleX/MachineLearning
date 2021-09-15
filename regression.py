from numpy import *
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from utils import tool


def lwlr(testPoint, xArr, yArr, k = 1.0):
    '''
    局部加权线性回归函数
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("this matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def ridgeRegres(testPoint, xArr, yArr, lam = 0.2):
    '''
    岭回归，与之类似的还有前向逐步回归
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("this matrix is singular")
        return
    ws = denom.I * (xMat.T * yMat)
    return testPoint * ws


def main():
    trainingset, traininglabels = tool.file2floatMatrix('ex0.txt', 2)
    yhat = lwlr(trainingset[0], trainingset, traininglabels, k = 1.0)
    print(yhat)
    yhat = lwlr(trainingset[0], trainingset, traininglabels, k = 0.001)
    print(yhat)

    lr = LinearRegression()
    lr.fit(trainingset, traininglabels)
    yhat = lr.predict([trainingset[0]])
    print(yhat)

    ridge = Ridge(alpha=1)
    ridge.fit(trainingset, traininglabels)
    yhat = ridge.predict([trainingset[0]])
    print(yhat)

    ridge = Ridge(alpha=0.001)
    ridge.fit(trainingset, traininglabels)
    yhat = ridge.predict([trainingset[0]])
    print(yhat)

    lasso = Lasso(alpha=1, max_iter=1000)
    lasso.fit(trainingset, traininglabels)
    yhat = lasso.predict([trainingset[0]])
    print(yhat)

    trainingset, traininglabels = tool.file2floatMatrix('abalone.txt', 8)
    yhat = ridgeRegres(trainingset[0], trainingset, traininglabels)
    print(yhat)


if __name__ == "__main__":
    main()