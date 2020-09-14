
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def simpleSNE():
    digits = load_digits()
    tsne = TSNE(random_state=42)
    digits_tsne = tsne.fit_transform(digits.data)

    plt.figure(figsize=(10, 10))
    plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
    plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
    for i in range(len(digits.data)):
        plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]), fontdict={'weight':'bold','size': 9})
    plt.xlabel("t-SNE feature 0")
    plt.xlabel("t-SNE feature 1")
    plt.show()


if __name__ == '__main__':
    simpleSNE()