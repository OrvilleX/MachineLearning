from numpy import *

# 朴素贝叶斯

def loadData():
    '''
    读取数据
    '''
    def textParse(bigstring):
        import re
        listoftokens = re.split('\W+', bigstring)
        return [tok.lower() for tok in listoftokens if len(tok) > 2]

    def setofwords2vec(voc, inputsest):
        returnvec = [0] * len(voc)
        for word in inputsest:
            if word in voc:
                returnvec[voc.index(word)] += 1
        return returnvec

    doclist = []
    classlist = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        doclist.append(wordList)
        classlist.append(1)

        wordList = textParse(open('email/ham/%d.txt' % i).read())
        doclist.append(wordList)
        classlist.append(0)
    vocabset = set([])
    for document in doclist:
        vocabset = vocabset | set(document)
    vocablist = list(vocabset)

    trainmat = []
    trainclasses = []
    for docindex in range(len(classlist)):
        trainmat.append(setofwords2vec(vocablist, doclist[docindex]))
        trainclasses.append(classlist[docindex])
    return trainmat, trainclasses


def trainBNO(trainmatrix, traincategory):
    '''
    朴素贝叶斯算法
    '''
    numtraindocs = len(trainmatrix)
    numwords = len(trainmatrix[0])
    padbusive = sum(traincategory) / float(numtraindocs)
    p0num = ones(numwords)
    p1num = ones(numwords)
    p0denom = 2.0
    p1denom = 2.0
    for i in range(numtraindocs):
        if traincategory[i] == 1:
            p1num += trainmatrix[i]
            p1denom += sum(trainmatrix[i])
        else:
            p0num += trainmatrix[i]
            p0denom += sum(trainmatrix[i])
    p1vect = log(p1num / p1denom)
    p0vect = log(p0num / p0denom)
    return p0vect, p1vect, padbusive


def classifynb(vec2classify, p0vec, p1vec, pclass1):
    '''
    计算结果
    '''
    p1 = sum(vec2classify * p1vec) + log(pclass1)
    p0 = sum(vec2classify * p0vec) + log(1.0 - pclass1)
    if p1 > p0:
        return 1
    else:
        return 0


def spamtest():
    trainingset, traininglabels = loadData()
    p0v, p1v, pab = trainBNO(array(trainingset[10:]), array(traininglabels[10:]))
    errorCount = 0
    for index in range(len(trainingset) - 40):
        if classifynb(array(trainingset[index]), p0v, p1v, pab) != traininglabels[index]:
            errorCount += 1
    print("the error rate is : ", float(errorCount) / (len(trainingset) - 40))


if __name__ == '__main__':
    spamtest()