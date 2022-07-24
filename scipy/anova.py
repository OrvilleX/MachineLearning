import numpy as np
from scipy import stats


def annovaTest():
    data = np.rec.array([
        ('Pat', 5), ('Pat', 4), ('Pat', 4), ('Pat', 3), ('Pat', 9), ('Pat', 4),
        ('Jack', 4), ('Jack', 8), ('Jack', 7), ('Jack', 5), ('Jack', 1),
        ('Jack', 5), ('Alex', 9), ('Alex', 8),
        ('Alex', 8), ('Alex', 10),('Alex', 5),
        ('Alex', 10)], dtype=[('Archer', '|U5'), ('Score', '<i8')])

    f, p = stats.ttest_ind(data[data['Archer'] == 'Alex'].Score,
                          data[data['Archer'] == 'Jack'].Score)
    print("ttest_ind test value:")
    print("F Value:", f)
    print("P Value:", p)

    f, p = stats.f_oneway(data[data['Archer'] == 'Pat'].Score,
                          data[data['Archer'] == 'Jack'].Score,
                          data[data['Archer'] == 'Alex'].Score)
    print("f_oneway test value:")
    print("F Value:", f)
    print("P Value:", p)

    f, p = stats.kruskal(data[data['Archer'] == 'Pat'].Score,
                          data[data['Archer'] == 'Jack'].Score,
                          data[data['Archer'] == 'Alex'].Score)
    print("kruskal test value:")
    print("F Value:", f)
    print("P Value:", p)

    result = stats.alexandergovern(data[data['Archer'] == 'Pat'].Score,
                          data[data['Archer'] == 'Jack'].Score,
                          data[data['Archer'] == 'Alex'].Score)
    print("alexandergovernTest test value:")
    print("F Value:", result.statistic)
    print("P Value:", result.pvalue)


if __name__ == '__main__':
    annovaTest()
