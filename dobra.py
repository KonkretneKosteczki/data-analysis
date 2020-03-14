import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

alpha = 0.05

df1 = pandas.read_table('data/4/Births.csv', sep=',', index_col=0)
print(f"births mean {df1.births.mean()}")
res1=stats.ttest_1samp(df1.births, 10000)
print(f"birth p-value: {res1.pvalue}")
if res1.pvalue > alpha:
    print("birth hypothesis is true")
else:
    print("birth hypothesis is false")
print('\n')

df2 = pandas.read_table('data/4/manaus.csv', sep=',', index_col=0)
print(f"manaus mean: {df2.manaus.mean()}")
res2=stats.ttest_1samp(df2.manaus, 0)
print(f"manaus p-value: {res2.pvalue}")
if res2.pvalue > alpha:
    print("manaus hypothesis is true")
else:
    print("manaus hypothesis is false")
print('\n')

df3 = pandas.read_table('data/4/quakes.csv', sep=',', index_col=0)
print(f"quakes mean: {df3.depth.mean()}")
res3=stats.ttest_1samp(df3.depth, 10000)
print(f"quakes p-value: {res3.pvalue}")
if res3.pvalue > alpha:
    print("quakes hypothesis is true")
else:
    print("quakes hypothesis is false")

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
sns.distplot(df1.births, color="blue", label="births number")
ax2 = fig.add_subplot(2,2,2)
sns.distplot(df2.manaus, color="blue", label="Manaus height")
ax3 = fig.add_subplot(2,2,3)
sns.distplot(df3.depth, color="blue", label="quakes depth")
plt.legend()
plt.show()

# todo:
# 2. Zwizualizować rozkłady na histogramie.
# 3. Zaznaczyć na wykresie punkt dotyczący badanej hipotezy.
# Należy zweryfikować normalność rozkładów, jeśli to konieczne.
