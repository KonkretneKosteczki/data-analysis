from sklearn import model_selection
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np


def train(model: SGDClassifier, train_data, train_labels, test_data, test_labels, total_epochs=1000):
    # using partial fit instead of fit in order to gather information on accuracy after every pass

    labels = []
    scores = []

    for i in range(total_epochs):
        if i == 0:
            model.partial_fit(train_data, train_labels, classes=np.unique(train_labels))
        else:
            model.partial_fit(train_data, train_labels)
        if (i + 1) % (total_epochs // 20) == 0:
            pred_labels = model.predict(test_data)
            scores.append(accuracy_score(test_labels, pred_labels))
            labels.append(i + 1)
            print("Epoch {0} score: {1}".format(i + 1, scores[-1]))

    return labels, scores


dataset_path = 'data/iris.data'
dataset_headers = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

df = pd.read_csv(dataset_path,
                 header=None,
                 names=dataset_headers)
X = df.iloc[:, :4]
y = df.iloc[:, 4]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

# 1. Przeprowadzić klasyfikację za pomocą wybranej techniki klasyfikacji (np. MLP, SVM, SGD, etc.)
sgd = SGDClassifier()
print("Training SGDClassifier")
training_data = train(sgd, X_train, y_train, X_test, y_test, 1000)

# 2. Wykonać wykres z przedstawioną skutecznością klasyfikatora
# (procent poprawnie sklasyfikowanych próbek) od liczby iteracji/epok.
plt.plot(*training_data)
plt.show()

# 3. Dokonać analizy głównych składowych (PCA) i zredukować liczbę cech do 2 składowych.
pca = PCA(n_components=2)
X_new = pca.fit_transform(X)

# 4. Ponownie przeprowadzić klasyfikację i porównać wyniki.
X_train2, X_test2, y_train2, y_test2 = model_selection.train_test_split(X_new, y)
sgd2 = SGDClassifier(max_iter=1000)
sgd2.fit(X_train2, y_train2)
y_pred2 = sgd2.predict(X_test2)
# res = sgd2.score(X_test2, y_test2)
res = accuracy_score(y_test2, y_pred2)
print(res)
