import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

alpha = 0.05
manaus_level_hypothesis=0

print(f"testing following hypothesis: \n"
      f"mean level of manaus river equal to: {manaus_level_hypothesis}\n"
      f"statistical significance: {alpha}")
df = pandas.read_table('data/4/manaus.csv', sep=',', index_col=0)
res = stats.ttest_1samp(df.manaus, manaus_level_hypothesis)
print(f"T-student test conducted on the data returned p-value: {res.pvalue}")
if res.pvalue > alpha:
    print("p-value is greater than the statistical significance, therefore the hypothesis is true")
else:
    print("p-value is smaller than the statistical significance, therefore the hypothesis is false")
print('\n')

sns.distplot(df.manaus, color="blue", label="Manaus level")
plt.axvline(0, color='k', linestyle='dashed', linewidth=1)
plt.legend()
plt.show()
