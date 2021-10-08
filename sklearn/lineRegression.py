from sklearn.linear_model import Ridge, Lasso, LinearRegression
from utils import tool


# 线性回归
def main():
    trainingset, traininglabels = tool.file2floatMatrix('../ex0.txt', 2)

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


if __name__ == "__main__":
    main()