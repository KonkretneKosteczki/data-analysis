import pandas as pd

# 1. Wszystkie wymagania na ocenę dostateczną oraz dodatkowo:
dataset_path = 'data/iris.data'
dataset_headers = ["sepal_length", "sepal_width", "petal_length","petal_width","class"]

df = pd.read_csv(dataset_path,
                 header=None,
                 names=dataset_headers)
X = df.iloc[:,:4] #training data
y = df.iloc[:,4]  #labels

# 2. Dokonać wyboru cech pozostawiając tylko 2, a odrzucając cechy o a) największej wariancji, b) najmniejszej wariancji

# print(df.var())
low_to_high_var_series = X.var().sort_values()
low_var_indices = low_to_high_var_series.head(2).keys()
high_var_indices = low_to_high_var_series.tail(2).keys()
df_low_var = X[low_var_indices]
df_high_var = X[high_var_indices]

# 3. Na zredukowanej liczbie cech przeprowadzić klasyfikację klasyfikatorem z wymagań na ocenę dostateczną.
# 4. Eksperyment powtórzyć wykorzystując do selekcji test niezależności chi^2.
# 5. Wykonać odpowiednie wykresy.
