import pandas

df = pandas.read_table('data/5/footballers.csv', sep=';')

#retaining only 2 positions
df = df.loc[df['Position'].isin(['Relief_Pitcher','Starting_Pitcher'])]
# df_factor = df.apply(lambda x: x.factorize()[0])
# print(df_factor)
correl = df.corr("spearman")
print(correl)