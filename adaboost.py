from numpy import *


def loadsimpdata():
    datamat = matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classlabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datamat, classlabels


def stumpclassify(datamatrix, dimen, threshval, threshineq):
    retarray = ones((shape(datamatrix)[0], 1))
    if threshineq == 'lt':
        retarray[datamatrix[:,dimen] <= threshval] = -1.0
    else:
        retarray[datamatrix[:,dimen] > threshval] = 1.0
    return retarray


def buildstump(dataarr, classlabels, d):
    datamatrix = mat(dataarr)
    labelmat = mat(classlabels).T
    m,n = shape(datamatrix)
    numsteps = 10.0
    beststump = {}
    bestclassest = mat(zeros((m,1)))
    minerror = inf
    for i in range(n):
        rangemin = datamatrix[:,i].min()
        rangemax = datamatrix[:,i].max()
        stepsize = (rangemax - rangemin)/numsteps
        for j in range(-1, int(numsteps) + 1):
            for inequal in ['lt', 'gt']:
                threshval = (rangemin + float(j) * stepsize)
                predictedvals = stumpclassify(datamatrix, i, threshval, inequal)
                errarr = mat(ones((m,1)))
                errarr[predictedvals == labelmat] = 0
                weightederror = d.T * errarr
                print('split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f' % (i, threshval, inequal, weightederror))
                if weightederror < minerror:
                    minerror = weightederror
                    bestclassest = predictedvals.copy()
                    beststump['dim'] = i
                    beststump['thresh'] = threshval
                    beststump['ineq'] = inequal
    return beststump, minerror, bestclassest


def adaboosttrainds(dataarr, classlabels, numit = 40):
    weakclassarr = []
    m = shape(dataarr)[0]
    d = mat(ones((m,1)) / 5)
    aggclassest = mat(zeros((m,1)))
    for i in range(numit):
        beststump, error, classest = buildstump(dataarr, classlabels, d)
        alpha = float(0.5*log((1.0 - error)/ max(error, 1e-16)))
        beststump['alpha'] = alpha
        weakclassarr.append(beststump)
        print("classest: ", classest.T)
        expon = multiply(-1 * alpha * mat(classlabels).T, classest)
        d = multiply(d, exp(expon))
        d = d/d.sum()
        aggclassest += alpha*classest
        print("aggclasses: ", aggclassest.T)
        aggerrors = multiply(sign(aggclassest) != mat(classlabels).T, ones((m,1)))
        errorrate = aggerrors.sum()/m
        print("total error: ", errorrate, "\n")
        if errorrate == 0.0:
            break
    return weakclassarr


def adaclassify(dattoclass, classifierarr):
    datamatrix = mat(dattoclass)
    m = shape(datamatrix)[0]
    aggclassest = mat(zeros((m,1)))
    for i in range(len(classifierarr)):
        classest = stumpclassify(datamatrix, classifierarr[i]['dim'], classifierarr[i]['thresh'], classifierarr[i]['ineq'])
        aggclassest += classifierarr[i]['alpha'] * classest
    return sign(aggclassest)


def main():
    datarr, labelarr = loadsimpdata()
    classifierarr = adaboosttrainds(datarr, labelarr, 30)
    ret = adaclassify([0, 0], classifierarr)
    print(ret)


if __name__ == '__main__':
    main()