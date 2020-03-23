import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr

df = pandas.read_table('data/5/footballers.csv', sep=';')
df_factor = df.apply(lambda x: x.factorize()[0])
correl = df.corr("spearman")
print(correl)