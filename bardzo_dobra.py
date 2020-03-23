import pandas

df = pandas.read_table('data/5/footballers.csv', sep=';')

# retaining only 2 positions and only important information (position and weight)
df = df.filter(["Weight(pounds)", "Position"]).loc[df['Position'].isin(['Relief_Pitcher', 'Starting_Pitcher'])]
df.Position = df.Position.factorize()[0]
print(df)
correl = df.corr("spearman")
print(correl)
