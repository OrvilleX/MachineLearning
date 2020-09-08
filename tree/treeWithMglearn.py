from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import graphviz

def treetest():
    cancer = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                        stratify=cancer.target,
                                                        random_state=42)
    tree = DecisionTreeClassifier(max_depth=4, random_state=0)
    tree.fit(x_train, y_train)

    # 导出决策树到文件
    export_graphviz(tree, out_file="../tree.dot", class_names=["malignant", "benign"],
                    feature_names=cancer.feature_names, impurity=False, filled=True)

    print("Accuracy on training set: {:.3f}".format(tree.score(x_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(tree.score(x_test, y_test)))

    # 读取决策树并显示
    with open("../tree.dot") as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph)
    graphviz.render('round-table.gv')


if __name__ == '__main__':
    treetest()