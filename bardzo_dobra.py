import pandas
import seaborn as sns
from matplotlib import pyplot as plt

import numpy as np
from scipy.stats import spearmanr

df = pandas.read_table('data/5/footballers.csv', sep=';')

# retaining only 2 positions and only important information (position and weight)
df = df.filter(["Weight(pounds)", "Position"]).loc[df["Position"].isin(['Relief_Pitcher', 'Starting_Pitcher'])]

#using scipy
alpha = 0.05

rho, pval = spearmanr(df["Weight(pounds)"], df["Position"])
print("statistical significance: ",alpha)
print("rho(correlation coefficient): ",rho)
print("pval: ", pval)
if pval>alpha:
    print("the results are statistically significant")
else:
    print("the results are not statistically significant")

#using pandas
factorizedPositions = df.Position.factorize()
factorizedPositionsLabels = factorizedPositions[1]
df.Position = factorizedPositions[0]

correl = df.corr(method="spearman")
print(correl)

# sns.pairplot(df, hue="Position") # very similar to the bottom but only displays the approximation
for i in range(len(factorizedPositions[1])):
    partialDataFrame = df.loc[df["Position"] == i]
    sns.distplot(partialDataFrame["Weight(pounds)"], label=f"{factorizedPositionsLabels[i]}", axlabel=False)
plt.legend()

# plt.figure(figsize=(5, 2))
# plt.scatter(df["Weight(pounds)"], df.Position)
# plt.yticks(np.arange(len(factorizedPositions[1])), factorizedPositions[1])

plt.show()
