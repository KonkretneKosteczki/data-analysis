import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

alpha = 0.05
manaus_level_hypothesis=0

print(f"Badanie następującej hipotezy: \n"
      f"Średnia wysokość rzeki w manaus jest na wysokości punktu arbitralnego(wynosi {manaus_level_hypothesis}).\n"
      f"poziom istotności statystycznej: {alpha}")
df = pandas.read_table('data/4/manaus.csv', sep=',', index_col=0)
stat, p_val = stats.ttest_1samp(df.manaus, manaus_level_hypothesis)
print(f"Wykonany test T-studenta dla danych testowych zwrócił p-wartość: {p_val}")

# df_mean = pandas.Series(len(df))
# for i in range(len(df)):
#     df_mean[i] = 0
# from scipy.stats import wilcoxon
# stat, p_val = wilcoxon(df['manaus'], df_mean)
# print("p_wartość wyznaczona testem wilcoxona: ", p_val)

if p_val > alpha:
    print("P-wartość jest wyższa niż przyjęty poziom istotności statystycznej. Nie ma podstaw do odrzucenia hipotezy zerowej.")
else:
    print("P-wartość jest niższa niż przyjęty poziom istotności statystycznej. Hipotezę odrzucamy, ponieważ różnice są istotne statystycznie.")
print('\n')

sns.distplot(df.manaus, color="blue", label="Manaus level")
plt.axvline(0, color='k', linestyle='dashed', linewidth=1)
plt.legend()
plt.show()
