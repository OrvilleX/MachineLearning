import tool
from numpy import *
from sklearn.neighbors import KNeighborsRegressor

# 监督学习——KNN算法

# KNN原始核心算法
def classify0(inX, dataSet: [], labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k) :
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 测试基于Numpy的KNN算法
def knnWithRawTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = tool.file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = tool.autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs) :
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is : %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]) :
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))


# 测试基于sklearn的KNN算法
def knnWithSklearnTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = tool.file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = tool.autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0

    reg = KNeighborsRegressor(n_neighbors=3)
    reg.fit(normMat[numTestVecs:m, :], datingLabels[numTestVecs:m])

    for i in range(numTestVecs):
        classifierResult = reg.predict([normMat[i, :]])
        print("the classifier came back with: %d, the real answer is : %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print("Test set R^2: {:.2f}".format(reg.score(normMat[0:numTestVecs,:], datingLabels[0:numTestVecs])))


if __name__ == "__main__":
    knnWithRawTest()
    knnWithSklearnTest()