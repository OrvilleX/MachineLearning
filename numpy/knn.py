from utils import tool
from numpy import *


# KNN原始核心算法
def classify0(inX, dataSet: [], labels, k):
    data_set_size = dataSet.shape[0]
    diff_mat = tile(inX, (data_set_size, 1)) - dataSet
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist_indicies[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


# 测试基于Numpy的KNN算法
def knnWithRawTest():
    ho_ratio = 0.1
    dating_data_mat, dating_labels = tool.file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = tool.autoNorm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m*ho_ratio)
    errorCount = 0.0
    for i in range(num_test_vecs) :
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 3)
        print("the classifier came back with: %d, the real answer is : %d" % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(num_test_vecs)))


if __name__ == "__main__":
    knnWithRawTest()