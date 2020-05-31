import pandas as pd
from sklearn import model_selection
from sklearn.feature_selection import chi2, SelectKBest

# 1. Wszystkie wymagania na ocenę dostateczną oraz dodatkowo:
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from task3.dostateczna import train, read_data, sgd_kwargs

def select_lowest_and_highest_variance_features(X):
    low_to_high_var_series = X.var().sort_values()
    low_var_indices = low_to_high_var_series.head(2).keys()
    high_var_indices = low_to_high_var_series.tail(2).keys()
    X_low_var = X[low_var_indices]
    X_high_var = X[high_var_indices]
    return X_low_var, X_high_var


def select_features_chi2(X, y):
    chi2selector = SelectKBest(score_func=chi2, k=2)
    chi2selector.fit_transform(X, y)
    selected_col_numbers = chi2selector.get_support(indices=True)
    X_chi2 = X.iloc[:, selected_col_numbers]
    return X_chi2

if __name__ == '__main__':
    df = read_data()
    X = df.iloc[:,:4] #training data
    y = df.iloc[:,4]  #labels
    # 2. Dokonać wyboru cech pozostawiając tylko 2, a odrzucając cechy o a) największej wariancji, b) najmniejszej wariancji

    X_low_var, X_high_var = select_lowest_and_highest_variance_features(X)

    # 3. Na zredukowanej liczbie cech przeprowadzić klasyfikację klasyfikatorem z wymagań na ocenę dostateczną.
    X_train_l, X_test_l, y_train_l, y_test_l = model_selection.train_test_split(X_low_var, y)
    sgd_l = SGDClassifier(**sgd_kwargs)
    print("\nTraining SGDClassifier for low variance features")
    training_data_l = train(sgd_l, X_train_l, y_train_l, X_test_l, y_test_l, 1000)
    plt.plot(*training_data_l)
    plt.show()

    X_train_h, X_test_h, y_train_h, y_test_h = model_selection.train_test_split(X_high_var, y)
    sgd_h = SGDClassifier(**sgd_kwargs)
    print("\nTraining SGDClassifier for high variance features")
    training_data_h = train(sgd_h, X_train_h, y_train_h, X_test_h, y_test_h, 1000)
    plt.plot(*training_data_h)
    plt.show()

    # 4. Eksperyment powtórzyć wykorzystując do selekcji test niezależności chi^2.
    X_chi2 = select_features_chi2(X, y)

    # 5. Wykonać odpowiednie wykresy.
    X_train_ch, X_test_ch, y_train_ch, y_test_ch = model_selection.train_test_split(X_chi2, y)
    sgd_ch = SGDClassifier(**sgd_kwargs)
    print("\nTraining SGDClassifier for high variance features")
    training_data_ch = train(sgd_ch, X_train_ch, y_train_ch, X_test_ch, y_test_ch, 1000)
    plt.plot(*training_data_ch)
    plt.show()
