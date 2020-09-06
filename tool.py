from numpy import *


def file2matrix(filename: str, column = 3):
    '''
    将文件内容转换为矩阵
    '''
    file = open(filename)
    arrayofLines = file.readlines()
    numberofLines = len(arrayofLines)
    returnMat = zeros((numberofLines, column))
    classLabelVector = []
    index = 0
    for line in arrayofLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0: column]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return array(returnMat), array(classLabelVector)


def autoNorm(dataset: array):
    '''
    归一化特征值
    '''
    minVals = dataset.min(0)
    maxVals = dataset.max(0)
    ranges = maxVals - minVals
    normDataset = zeros(shape(dataset))
    m = dataset.shape[0]
    normDataset = dataset - tile(minVals, (m, 1))
    normDataset = normDataset / tile(ranges, (m, 1))
    return normDataset, ranges, minVals


def file2floatMatrix(filename: str, column = 3):
    '''
    读取文件内容并以浮点数值组织矩阵
    '''
    file = open(filename)
    trainingset = []
    traininglabels = []
    for line in file.readlines():
        currline = line.strip().split('\t')
        linearr = []
        for i in range(column):
            linearr.append(float(currline[i]))
        trainingset.append(linearr)
        traininglabels.append(float(currline[-1]))
    return array(trainingset), array(traininglabels)