from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import itertools


def subject_accuracy(matrix):
    # assume the maximum value in a particular row to be correct
    label_count = len(matrix)
    correct = 0
    wrong = 0
    for i in range(label_count):
        maxima = np.argmax(matrix[i])
        print(matrix[i][maxima])
        correct += matrix[i][maxima]
        wrong += sum(np.delete(matrix[i], maxima))
    print("Subject Accuracy {0}".format(correct/(correct+wrong)))


def rearrange_rows(matrix, pred_labels):
    # assume the maximum value in a particular row to be correct
    # and rearrange rows accordingly
    label_count = len(matrix)
    new_matrix = np.empty((label_count, label_count))
    new_labels = np.empty(pred_labels.size)
    for i in range(label_count):
        maxima = np.argmax(matrix[i])
        new_matrix[maxima] = matrix[i]
        new_labels[maxima] = pred_labels[i]
    return new_matrix, new_labels


def calculate_accuracy(pred_labels, true_labels):
    unique_true_labels = np.unique(true_labels)
    unique_pred_labels = np.unique(pred_labels)
    matrix = np.zeros((unique_true_labels.size, unique_pred_labels.size))

    labels_dicts = dict(zip(unique_true_labels, unique_pred_labels))
    for i in range(len(pred_labels)):
        matrix[pred_labels[i]][labels_dicts.get(true_labels[i])] += 1

    subject_accuracy(matrix)
    matrix, unique_pred_labels = rearrange_rows(matrix, unique_pred_labels)

    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(unique_true_labels))
    plt.xticks(tick_marks, unique_true_labels)
    plt.yticks(tick_marks, unique_pred_labels)

    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], '.0f'), horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')


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

calculate_accuracy(y_pred, Y.values)

# plt.show()
