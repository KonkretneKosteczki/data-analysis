from task2.dostateczna import percent, read_data, regression_curve_before, mean_imputation, compare_data, \
    regression_curve_after

# Wybrać zbiór danych do analizy, korzystając z dowolnego repozytorium danych (UCI Repository, Kaggle, inne). W zbiorze powinny występować brakujące dane, jednak nie powinno być ich zbyt wiele (około 5%-10%). Proszę również zwrócić uwagę, że dalsza część poleceń uwzględnia regresję liniową, która jest stosowana dla skal ilościowych (i zasadniczo zakłada zgodność z rozkładem normalnym). Proszę zatem zadbać, żeby parametry, które podlegają regresji liniowej miały charakterystykę pozwalającą na zaobserwowanie różnic (czyli chociażby charakteryzowały się zauważalną wariancją).
x_column = "TOEFL_Score"
y_column = "CGPA"

# Wczytać dane z brakami, policzyć jaki procent danych zawiera braki.
data = read_data()
percent(data)
# Wyznaczyć krzywą regresji dla danych bez braków.
model, r_sq = regression_curve_before(data, x_column, y_column)
# Uzupełnić braki metodą "mean imputation"
filled = mean_imputation(data)
# Porównać charakterystykę zbiorów przed i po imputacji (średnia, odchylenie standardowe, kwartyle).
compare_data(data, filled, x_column, y_column)
# Wyznaczyć krzywą regresji dla danych po imputacji. Porównać jak zmieniły się parametry krzywej
regression_curve_after(filled, model, r_sq, x_column, y_column)



