import statistics
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def read_data():
    raw_data = pd.read_csv(open("data/admission.csv"), delimiter=";")
    data = raw_data.drop(["Serial No."], axis=1)
    return data

def percent(data_frame):
    count = data_frame.count()  # count of data in each column
    print("Ilosc danych w kolumnach:")
    print(count)

    # missing fields = number of columns * number of rows - sum of fields in each row
    total_rows = len(data_frame.index)
    missing_fields = len(count) * total_rows - sum(count)

    per = (missing_fields / total_rows) * 100
    print("Procent danych z brakami: ", per)
    print()
    return per


def create_graph(coefficient, intercept, x_data, y_data, graph_label, x_label=None, y_label=None):
    if x_label is None or y_label is None:
        raise ValueError("x_label and y_label cannot be None")
    min_x = min(x_data[x_label])
    max_x = max(x_data[x_label])
    x = np.linspace(min_x, max_x, 100)
    y = coefficient * x + intercept
    plt.plot(x, y, "-r")
    plt.plot(x_data, y_data, "ro")

    plt.title(graph_label)
    plt.xlabel(x_label, color="#1C2833")
    plt.ylabel(y_label, color="#1C2833")
    plt.grid()
    plt.show()


def regression_curve_before(data, x_column, y_column, graph_name="Graph of linear regression before"):
    data_without_nans = data.dropna()  # removes rows with missing data

    x_data = data_without_nans[[x_column]]
    y_data = data_without_nans[y_column]
    # print(data_without_nans.corr()[y_column])
    model = LinearRegression().fit(x_data, y_data) # takes: X{array-like, sparse matrix} of shape (n_samples, n_features);yarray-like of shape (n_samples,) or (n_samples, n_targets)
    r_sq = model.score(x_data, y_data)
    print("Parametry krzywej regresji dla danych bez braków:")
    print("R^2:", r_sq)
    print("intercept:", model.intercept_)
    print("slope:", model.coef_)
    print()
    create_graph(model.coef_, model.intercept_, x_data, y_data, graph_name, x_column, y_column)
    return model, r_sq


def mean_imputation(data):
    filled = pd.DataFrame()
    for (columnName, columnData) in data.iteritems():
        filled[columnName] = columnData.fillna(columnData.mean())

    return filled


def compare_data(data_before, data_after, x_column, y_column, description=None):
    def calc_mean_dev_quantile(data):
        return [data[y_column].mean(), statistics.stdev(data[y_column]), data[y_column].quantile([0.25, 0.5, 0.75])]

    mean, st_dev, quantile = calc_mean_dev_quantile(data_before.dropna())
    mean_after, st_dev_after, quantile_after = calc_mean_dev_quantile(data_after)

    if description is not None:
        print(description)
    print("Srednia przed: ", mean)
    print("Srednia po: ", mean_after)
    print("Roznica: ", mean_after - mean)
    print()

    print("Odchylenie standardowe przed: ", st_dev)
    print("Odchylenie standardowe po: ", st_dev_after)
    print("Roznica: ", st_dev_after - st_dev)
    print()

    print("Kwartyle przed: ")
    print(quantile)
    print("Kwartyle po: ")
    print(quantile_after)
    print("Roznica: ")
    print(quantile_after - quantile)
    print()

def regression_curve_after(data, model, r_sq, x_column, y_column, graph_text="Graph of linear regression after"):
    x_data = data[[x_column]]
    y_data = data[y_column]
    model_2 = LinearRegression().fit(x_data, y_data)
    r_sq_2 = model.score(x_data, y_data)
    print("R^2 po imputacji:", r_sq_2)
    print("intercept po imputacji:", model_2.intercept_)
    print("slope po imputacji:", model_2.coef_)
    print()
    print("Różnica R^2:", (r_sq_2 - r_sq))
    print("Różnica intercept:", model_2.intercept_ - model.intercept_)
    print("Różnica slope:", model_2.coef_ - model.coef_)
    print()
    create_graph(model_2.coef_, model_2.intercept_, x_data, y_data, graph_text, x_column,
                 y_column)

if __name__ == '__main__':
    # Wybrać zbiór danych do analizy, korzystając z dowolnego repozytorium danych (UCI Repository, Kaggle, inne). W zbiorze powinny występować brakujące dane, jednak nie powinno być ich zbyt wiele (około 5%-10%). Proszę również zwrócić uwagę, że dalsza część poleceń uwzględnia regresję liniową, która jest stosowana dla skal ilościowych (i zasadniczo zakłada zgodność z rozkładem normalnym). Proszę zatem zadbać, żeby parametry, które podlegają regresji liniowej miały charakterystykę pozwalającą na zaobserwowanie różnic (czyli chociażby charakteryzowały się zauważalną wariancją).
    y_column = "TOEFL_Score"
    x_column = "GRE_Score"

    # Wczytać dane z brakami, policzyć jaki procent danych zawiera braki.
    data = read_data()
    percent(data)
    # Wyznaczyć krzywą regresji dla danych bez braków.
    model, r_sq = regression_curve_before(data, x_column, y_column)
    # Uzupełnić braki metodą "mean imputation"
    filled = mean_imputation(data)
    # Porównać charakterystykę zbiorów przed i po imputacji (średnia, odchylenie standardowe, kwartyle).
    compare_data(data, filled, x_column, y_column, description="Porównanie charakterystyki zbiorów z brakami i po imputacji 'mean imputation'")
    # Wyznaczyć krzywą regresji dla danych po imputacji. Porównać jak zmieniły się parametry krzywej
    regression_curve_after(filled, model, r_sq, x_column, y_column)