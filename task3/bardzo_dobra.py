from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd


dataset_path = 'data/iris.data'
dataset_headers = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv(dataset_path,
                 header=None,
                 names=dataset_headers)
X = df.iloc[:, :4]
Y = df.iloc[:, 4]

pca = PCA(n_components=2)
X = pca.fit_transform(X)

y_pred = KMeans(n_clusters=np.unique(Y).size, random_state=100).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("PCA KMeans")

plt.show()