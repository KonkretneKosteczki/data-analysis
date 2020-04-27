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
    res= data.sort_values("GRE_Score").fillna(method="ffill") #LOCF
    for a in res[y_column]:
        print(a)
    return res

def regression_curve_imputation(adata, model):
    data = adata.copy()
    data.loc[(pd.isnull(data[y_column])), y_column] = data[x_column] * model.coef_ + model.intercept_
    return data


if __name__ == '__main__':
    # 1. Wszystkie wymagania na ocenę dostateczną.
    initial_data = read_data()
    y_column = "TOEFL_Score" #missing data
    x_column = "GRE_Score"
    model, r_sq = regression_curve_before(initial_data, x_column, y_column)
    # 2. Uzupełnić braki metodą interpolacji, hot-deck, oraz wartościami uzyskanymi z krzywej regresji wyznaczonej na podstawie danych bez braków.
    filled_interpolation = interpolation_imputation(initial_data)
    filled_hot_deck = hot_deck_imputation(initial_data)
    filled_regression=regression_curve_imputation(initial_data, model)
    # 3. Porównać charakterystyki zbiorów przed i po imputacji (średnia, odchylenie standardowe, kwartyle).
    compare_data(initial_data, filled_interpolation, x_column, y_column, description="Porównanie charakterystyki zbiorów z brakami i po imputacji metodą INTERPOLACJI:")
    compare_data(initial_data, filled_hot_deck, x_column, y_column, description="Porównanie charakterystyki zbiorów z brakami i po imputacji metodą HOT-DECK:")
    compare_data(initial_data, filled_regression, x_column, y_column, description="Porównanie charakterystyki zbiorów z brakami i po imputacji wartościami z KRZYWEJ REGRESJI:")
    # 4. Wyznaczyć krzywą regresji dla danych po każdej imputacji. Porównać jak zmieniły się parametry krzywych.
    print("Parametry krzywej regresji po imputacji metodą INTERPOLACJI:")
    regression_curve_after(filled_interpolation, model, r_sq, x_column, y_column, graph_text="Graph of linear regression after interpolation imputation")
    print("Parametry krzywej regresji po imputacji metodą HOT-DECK:")
    regression_curve_after(filled_hot_deck, model, r_sq, x_column, y_column, graph_text="Graph of linear regression after hot-deck imputation")
    print("Parametry krzywej regresji po imputacji wartościami z KRZYWEJ REGRESJI:")
    regression_curve_after(filled_regression, model, r_sq, x_column, y_column, graph_text="Graph of linear regression after regression curve imputation")