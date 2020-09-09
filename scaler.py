from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
import mglearn


"""
为了快速了解如何利用sklearn进行数据的预处理，读者可以通过使用 mglearn.plots.plot_scaling 来显示
常用的四种缩放方式，对于每种缩放的解释如下：

StandardScaler：确保每个特征的平均值为0、方差为1，使所有特征都位于同一量级，但是这种缩放不能保证特征
任何特定的最大值和最小值。

RobustScaler：确保每个特征的中位数与四分位数，这样会忽略与其他点有很大不同的数据点。

MinMaxScaler：移动数据点，使所有特征刚好位于0到1之间。

Normalizer：对每个数据点进行缩放，使得特征向量的欧式长度等于1，其实就是将数据点投射到半径为1的圆上。
"""


def testMinMaxScaler():
    cancer = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
    scaler = MinMaxScaler()

    """
    使用fit方法拟合缩放器，对于当前fit将计算训练集中每隔二特征的最大值和最小值
    """
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train) # 也可以使用快捷方式 fit_transform

    print("transformed shape: {}".format(x_train_scaled.shape))
    print("per-feature minimum before scaling:\n {}".format(x_train.min(axis=0)))
    print("pre-feature maximum before scaling:\n {}".format(x_train.max(axis=0)))
    print("pre-feature minimum after scaling:\n {}".format(x_train_scaled.min(axis=0)))
    print("pre-feature maximum after scaling:\n {}".format(x_train_scaled.max(axis=0)))

    """
    对测试数据进行变换，这里需要注意测试数据不能单独使用fit重新训练然后在进行变化，这样的
    使用会导致变换后的数据与原数据存在巨大差异。
    """

    x_test_scaled = scaler.transform(x_test)

    print("per-feature minimum after scaling:\n {}".format(x_test_scaled.min(axis=0)))
    print("pre-feature maximum after scaling:\n {}".format(x_test_scaled.max(axis=0)))

    scaler2 = StandardScaler()
    scaler2.fit(x_train)
    x_train_scaled2 = scaler2.transform(x_train)
    x_test_scaled2 = scaler2.transform(x_test)

    scaler3 = RobustScaler()
    scaler3.fit(x_train)
    x_train_scaled3 = scaler3.transform(x_train)
    x_test_scaled3 = scaler3.transform(x_test)

    scaler4 = Normalizer()
    scaler4.fit(x_train)
    x_train_scaled4 = scaler4.transform(x_train)
    x_test_scaled4 = scaler4.transform(x_test)




if __name__ == '__main__':
    mglearn.plots.plot_scaling()
    i = {}