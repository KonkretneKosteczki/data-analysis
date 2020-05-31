from sklearn import model_selection
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plt
dataset_path = 'data/iris.data'
dataset_headers = ["sepal_length", "sepal_width", "petal_length","petal_width","class"]

df = pd.read_csv(dataset_path,
                 header=None,
                 names=dataset_headers)
X = df.iloc[:,:4]
y = df.iloc[:,4]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y)

#1. Przeprowadzić klasyfikację za pomocą wybranej techniki klasyfikacji (np. MLP, SVM, SGD, etc.)
sgd = SGDClassifier(max_iter=1000)
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)
# res = sgd.score(X_test, y_test)
res = accuracy_score(y_test, y_pred)
print(res)
# 2. Wykonać wykres z przedstawioną skutecznością klasyfikatora (procent poprawnie sklasyfikowanych próbek) od liczby iteracji/epok.
train_sizes_abs, train_scores, test_scores = learning_curve(sgd, X_train, y_train)
# todo: wyświetlić krzywą

# 3. Dokonać analizy głównych składowych (PCA) i zredukować liczbę cech do 2 składowych.
pca = PCA(n_components=2)
X_new = pca.fit_transform(X)

# 4. Ponownie przeprowadzić klasyfikację i porównać wyniki.
X_train2, X_test2, y_train2, y_test2 = model_selection.train_test_split(X_new,y)
sgd2 = SGDClassifier(max_iter=1000)
sgd2.fit(X_train2, y_train2)
y_pred2 = sgd2.predict(X_test2)
# res = sgd2.score(X_test2, y_test2)
res = accuracy_score(y_test2, y_pred2)
print(res)
