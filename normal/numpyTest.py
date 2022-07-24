import numpy as np
import matplotlib.pyplot as plt


def normalTest():
    s = np.random.normal(size=1000)
    bins = np.arange(-4, 5)
    histogram = np.histogram(s, bins=bins, density=True)[0]
    """
    累计直方图，用于将数据根据要求进行计算，以提供便于直方图显示的数据形式
    bins指定统计的区间个数
    density为True时，返回每个区间的概率密度；为False，返回每个区间中元素的个数
    """
    bins = 0.5 * (bins[1:] + bins[:-1])
    plt.plot(bins, histogram)


if __name__ == '__main__':
    normalTest()