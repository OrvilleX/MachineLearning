from numpy import *
from sklearn.naive_bayes import MultinomialNB
from raw import bayes as bayes

'''
其他朴素贝叶斯还有
GaussianNB 适用于通用场景
BernoulliNB 适用于属性代表是否关系
MultinomialNB 适用于属性用于计数场景

通过max_depth可以控制树的深度
'''

def mnbtest():
    errorCount = 0
    trainingset, traininglabels = bayes.loadData()
    clf = MultinomialNB()
    clf.fit(array(trainingset[10:]), array(traininglabels[10:]))
    for index in range(len(trainingset) - 20):
        isok = clf.predict(array([trainingset[index]]))
        if isok != traininglabels[index]:
            errorCount += 1
    print("the error rate is : ", float(errorCount) / (len(trainingset) - 20))
    print("the score is : ", clf.score(array(trainingset[:10]), array(traininglabels[:10])))


if __name__ == '__main__':
    mnbtest()