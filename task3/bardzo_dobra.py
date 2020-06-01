from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import itertools

from task3.dobra import select_lowest_and_highest_variance_features, select_features_chi2
from task3.dostateczna import read_data, select_PCA_features


def subject_accuracy(matrix, title):
    # assume the maximum value in a particular row to be correct
    label_count = len(matrix)
    correct = 0
    wrong = 0
    for i in range(label_count):
        maxima = np.argmax(matrix[i])
        correct += matrix[i][maxima]
        wrong += sum(np.delete(matrix[i], maxima))
    print(title + " Subject Accuracy {0}".format(correct/(correct+wrong)))


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


def calculate_accuracy(pred_labels, true_labels, title):
    unique_true_labels = np.unique(true_labels)
    unique_pred_labels = np.unique(pred_labels)
    matrix = np.zeros((unique_true_labels.size, unique_pred_labels.size))

    labels_dicts = dict(zip(unique_true_labels, unique_pred_labels))
    for i in range(len(pred_labels)):
        matrix[pred_labels[i]][labels_dicts.get(true_labels[i])] += 1

    subject_accuracy(matrix, title)
    matrix, unique_pred_labels = rearrange_rows(matrix, unique_pred_labels)

    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title + " Confusion Matrix")
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


def analyse_clusters(X, Y, title):
    plt.figure()
    y_pred = KMeans(n_clusters=np.unique(Y).size, random_state=100).fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title(title)
    # 3. Sprawdzić jakość utworzonych skupień za pomocą wybranego przez siebie kryterium.
    # Przeanalizować wyniki, wykreślić wykresy, jeśli są wskazane.
    calculate_accuracy(y_pred, Y.values, title)


# 2. Dokonać analizy skupień przy pomocy wybranego algorytmu
# (np. k-means, meanshift, spectral clustering, etc.) używając cech wyznaczonych przez PCA
# w części zadania na ocenę dostateczną oraz cech wyznaczonych w wymaganiach na ocenę dobrą.


df = read_data()
X = df.iloc[:, :4]
Y = df.iloc[:, 4]

## dostateczna
X_pca = select_PCA_features(X)
analyse_clusters(X_pca, Y, "PCA KMeans")

## dobra
X_low_var, X_high_var = select_lowest_and_highest_variance_features(X)
X_chi2_h, X_chi2_l = select_features_chi2(X, Y)
analyse_clusters(X_low_var.values, Y, "Low Var KMeans")
analyse_clusters(X_high_var.values, Y, "High Var KMeans")
analyse_clusters(X_chi2_h.values, Y, "Chi2 high correalation KMeans")
analyse_clusters(X_chi2_l.values, Y, "Chi2 loose correalation KMeans")

plt.show()
