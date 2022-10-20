import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_embeddings(embeddings,y):
    X = []
    Y = []
    for i, v in enumerate(y):
        X.append(i)
        Y.append(v)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(embeddings)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i].item(), [])
        color_idx[Y[i].item()].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()