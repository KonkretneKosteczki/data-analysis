import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv('data/3/abalone.data',
                 header=None,
                 names=["Sex", "Length", "Diameter", "Height", "Whole weight",
                        "Shucked weight", "Viscera weight", "Shell weight", "Rings"])
of = pd.DataFrame()
of["mean"] = df.mean(axis=0)
of["min"] = df.min(axis=0)
of["max"] = df.max(axis=0)

print(df.Sex.mode())
print(of)

sns.distplot(df.Length, color="blue", label="Length", axlabel=False)
sns.distplot(df.Diameter, color="red", label="Diameter", axlabel=False)

plt.legend()
plt.show()
