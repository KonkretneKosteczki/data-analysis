import math
import numpy as np
import pandas as pd

from task2.dostateczna import read_data, regression_curve_before, compare_data, regression_curve_after


def interpolation_imputation(data):
    filled = pd.DataFrame()
    for (columnName, columnData) in data.iteritems():
        filled[columnName] = columnData.fillna(columnData.interpolate())
    return filled

def hot_deck_imputation(data):
    return data.fillna(method="ffill") #LOCF

def regression_curve_imputation(adata, model):
    data = adata.copy()
    data.loc[(pd.isnull(data[x_column])), x_column] = data[y_column]/model.coef_-model.intercept_
    return data


if __name__ == '__main__':
    # 1. Wszystkie wymagania na ocenę dostateczną.
    initial_data = read_data()
    x_column = "TOEFL_Score" #missing data
    y_column = "CGPA"
    model, r_sq = regression_curve_before(initial_data, x_column, y_column)
    # 2. Uzupełnić braki metodą interpolacji, hot-deck, oraz wartościami uzyskanymi z krzywej regresji wyznaczonej na podstawie danych bez braków.
    filled_interpolation = interpolation_imputation(initial_data)
    filled_hot_deck = hot_deck_imputation(initial_data)
    filled_regression=regression_curve_imputation(initial_data, model)
    # 3. Porównać charakterystyki zbiorów przed i po imputacji (średnia, odchylenie standardowe, kwartyle).
    print("INTERPOLATION")
    compare_data(initial_data, filled_interpolation, x_column, y_column)
    print("HOT-DECK")
    compare_data(initial_data, filled_hot_deck, x_column, y_column)
    print("REGRESSION")
    compare_data(initial_data, filled_regression, x_column, y_column)
    # 4. Wyznaczyć krzywą regresji dla danych po każdej imputacji. Porównać jak zmieniły się parametry krzywych.
    print("INTERPOLATION")
    regression_curve_after(filled_interpolation, model, r_sq, x_column, y_column, graph_text="Graph of linear regression after interpolation imputation")
    print("HOT-DECK")
    regression_curve_after(filled_hot_deck, model, r_sq, x_column, y_column, graph_text="Graph of linear regression after hot-deck imputation")
    print("REGRESSION")
    regression_curve_after(filled_regression, model, r_sq, x_column, y_column, graph_text="Graph of linear regression after regression curve imputation")