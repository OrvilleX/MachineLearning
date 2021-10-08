from sklearn.linear_model import LogisticRegression
from utils import tool


# 逻辑回归算法
def testLogistic():
    """
    基于sklearn的Logistic回归
    """
    logreg = LogisticRegression(C=1)
    trainingset, traininglabels = tool.file2floatMatrix('../horseColicTraining.txt', 21)
    testset, testlabels = tool.file2floatMatrix('../horseColicTest.txt', 21)
    logreg.fit(trainingset, traininglabels)
    print("logreg.coef_: {}".format(logreg.coef_))
    print("logreg.intercept_: {}".format(logreg.intercept_))
    print("Training set score: {:.2f}".format(logreg.score(trainingset, traininglabels)))
    print("Test set score: {:.2f}".format(logreg.score(testset, testlabels)))


if __name__ == '__main__':
    testLogistic()
