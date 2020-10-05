import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def pandasMain():
    data = pd.read_csv(
        'adult.data', header=None, index_col=False,
        names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race', 'gender',
               'capital-gain', 'capitaal-loss', 'hours-per-week', 'native-country',
               'income'])
    # 选取需要的数据
    data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']]
    # 查看特定数据出现各类值次数
    print(data.gender.value_counts())
    # 对数据进行one-hot处理
    print("Original features:\n", list(data.columns), "\n")
    data_dummies = pd.get_dummies(data)
    print("Features after get_dummies:\n", list(data_dummies.columns))

    # 选择需要训练的数据
    features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']
    x = features.values
    y = data_dummies['income_ >50K'].values
    print("x.shape: {} y.shape: {}".format(x.shape, y.shape))

    # 进行模型训练
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    print('Test score: {:.2f}'.format(logreg.score(x_test, y_test)))


if __name__ == '__main__':
    # 基于pandas的one-hot
    pandasMain()