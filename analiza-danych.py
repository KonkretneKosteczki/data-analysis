import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv('data/3/abalone.data',
                 header=None,
                 names=["Sex", "Length", "Diameter", "Height", "Whole weight",
                        "Shucked weight", "Viscera weight", "Shell weight", "Rings"])
#quantitive
of = pd.DataFrame()
of["median"] = df.median(axis=0)
of["min"] = df.min(axis=0)
of["max"] = df.max(axis=0)
print(of)



#qualitive
print('\nMode for qualitive attribute: ')
print(df.Sex.mode())

#getting correlations
correl = df.corr()
#eliminating correlating same labels e.g. Length with Length
for i in correl.index:
    for j in correl.columns:
        if i==j:
            correl.loc[i,j] = 0
corr_val1 = correl.idxmax().index[0]
corr_val2 = correl.idxmax()[0]
print(f"the most correlated values are: {corr_val1} and {corr_val2}")


sns.distplot(df[corr_val1], color="blue", label=f"{corr_val1}", axlabel=False)
sns.distplot(df[corr_val2], color="red", label=f"{corr_val2}", axlabel=False)

plt.legend()
plt.show()
