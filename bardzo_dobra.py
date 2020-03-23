import pandas
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

df = pandas.read_table('data/5/footballers.csv', sep=';')

# retaining only 2 positions and only important information (position and weight)
df = df.filter(["Weight(pounds)", "Position"]).loc[df["Position"].isin(['Relief_Pitcher', 'Starting_Pitcher'])]
factorizedPositions = df.Position.factorize()
df.Position = factorizedPositions[0]

correl = df.corr(method="spearman")
print(correl)

df1 = df.loc[df['Position'] == 0]
df2 = df.loc[df['Position'] == 1]
sns.distplot(df1["Weight(pounds)"], color="blue", label=f"{factorizedPositions[0]}", axlabel=False)
sns.distplot(df2["Weight(pounds)"], color="red", label=f"{factorizedPositions[1]}", axlabel=False)


plt.figure(figsize=(5, 2))
plt.scatter(df["Weight(pounds)"], df.Position)
plt.yticks(np.arange(len(factorizedPositions[1])), factorizedPositions[1])

plt.show()
