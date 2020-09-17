from numpy import *


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


# 根据最小支持度从D中搜索符合Ck项集的数据
def scanD(D, Ck, minSupport):
    ssCnt = {}
    numItems = float(len(D))
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.__contains__(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


def aprioriGen(lk, k):
    retList = []
    lenlk = len(lk)
    for i in range(lenlk):
        for j in range(i+1, lenlk):
            l1 = list(lk[i])[:k-2]
            l2 = list(lk[j])[:k-2]
            l1.sort()
            l2.sort()
            if l1 == l2:
                retList.append(lk[i] | lk[j])
    return retList


# Apriori核心算法
def apriori(dataSet, minSupport = 0.5):
    c1 = createC1(dataSet)
    d = list(map(set, dataSet))
    l1, supportData = scanD(d, c1, minSupport)
    l = [l1]
    k = 2
    while (len(l[k-2]) > 0):
        ck = aprioriGen(l[k-2], k)
        lk, supk = scanD(d, ck, minSupport)
        supportData.update(supk)
        l.append(lk)
        k += 1
    return l, supportData


def calcConf(freqSet, h, supportData, brl, minConf = 0.7):
    prundh = []
    for conseq in h:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            brl.append((freqSet-conseq, conseq, conf))
            prundh.append(conseq)
    return prundh


def rulesFromConseq(freqSet, h, supportData, brl, minConf = 0.7):
    m = len(h[0])
    if (len(freqSet) > (m + 1)):
        hmpl = aprioriGen(h, m+1)
        hmpl = calcConf(freqSet, hmpl, supportData, brl, minConf)
        if (len(hmpl) > 1):
            rulesFromConseq(freqSet, hmpl, supportData, brl, minConf)


# 根据置信度查询关联规则
def generateRules(l, supportData, minConf = 0.7):
    bigRuleList = []
    for i in range(1, len(l)):
        for freqSet in l[i]:
            hl = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, hl, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, hl, supportData, bigRuleList, minConf)
    return bigRuleList


if __name__ == '__main__':
    dataset = loadDataSet()
    l, supportData = apriori(dataset)
    rules = generateRules(l, supportData, minConf= .5)
    print(rules)