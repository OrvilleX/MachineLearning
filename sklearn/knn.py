from utils import tool
from sklearn.neighbors import KNeighborsRegressor


# 测试基于sklearn的KNN算法
def knnWithSklearnTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = tool.file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = tool.autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0

    reg = KNeighborsRegressor(n_neighbors=3)
    reg.fit(normMat[numTestVecs:m, :], datingLabels[numTestVecs:m])

    for i in range(numTestVecs):
        classifierResult = reg.predict([normMat[i, :]])
        print("the classifier came back with: %d, the real answer is : %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print("Test set R^2: {:.2f}".format(reg.score(normMat[0:numTestVecs, :], datingLabels[0:numTestVecs])))


if __name__ == "__main__":
    knnWithSklearnTest()