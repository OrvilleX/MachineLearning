from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import mglearn
import numpy as np

"""
NMF非负矩阵分解
"""

def simpleNMF():
    # 提取每人50图片
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    x_people = people.data[mask]
    y_people = people.target[mask]
    x_people = x_people / 255 # 将灰度固定在0~1之间
    x_train, x_test, y_train, y_test = train_test_split(x_people, y_people, stratify=y_people, random_state=0)

    nmf = NMF(n_components=15, random_state=0)
    nmf.fit(x_train)
    x_train_nmf = nmf.transform(x_train)
    x_test_nmf = nmf.transform(x_test)

    fix, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks':(), 'yticks': ()})

    for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
        ax.imshow(component.reshape(image_shape))
        ax.set_title("{}. component".format(i))
    plt.show()


if __name__ == '__main__':
    simpleNMF()