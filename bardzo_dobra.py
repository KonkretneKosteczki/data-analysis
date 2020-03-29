import pandas
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import norm


from scipy.stats import spearmanr

df = pandas.read_table('data/5/footballers.csv', sep=';')

# retaining only 2 positions and only important information (position and weight)
df = df.filter(["Weight(pounds)", "Position"]).loc[df["Position"].isin(['Relief_Pitcher', 'Starting_Pitcher'])]

#using scipy
alpha = 0.05

rho, pval = spearmanr(df["Weight(pounds)"], df["Position"])
print("Przyjęty poziom istotnośći statystycznej: ", alpha)
print("Współczynnik korelacji wyznaczony metodą rank spearmana: ", rho)
print("P-wartość: ", pval)

if rho > 0.3:
    print("Występuje zależność między badanymi cechami. Współczynnik korelacji jest wysoki.")
else:
    print("Brak zależności między badanymi cechami. Współczynnik korelacji jest niski.")

if pval>alpha:
    print("Wynik jest mało istotny statystycznie. P-wartość jest wyższa niż przyjęty poziom istotności statystycznej.")
else:
    print("Wynik ma dużą istotność statystyczną. P-wartość jest niższa niż przyjęty poziom istotności statystycznej.")

partialDataFrame = df.loc[df["Position"] == 'Relief_Pitcher']
color1 = sns.color_palette("hls", 2)[0]
sns.distplot(partialDataFrame["Weight(pounds)"], label='Relief_Pitcher', kde=False,
                 color=color1, fit=norm, fit_kws={"color": color1})

partialDataFrame = df.loc[df["Position"] == 'Starting_Pitcher']
color2 = sns.color_palette("hls", 2)[1]
sns.distplot(partialDataFrame["Weight(pounds)"], label='Starting_Pitcher',  kde=False,
                 color=color2, fit=norm, fit_kws={"color": color2})

plt.legend()
plt.show()
