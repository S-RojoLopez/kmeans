# %%
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator

df = pd.read_csv("ulabox_orders_with_categories_partials_2017.csv")

# ================================
# Determina el número K óptimo
# ================================

dfp = df.drop(columns=["order", "customer"])

ssd = []
ks = range(1,11)
for k in range(1,11):
    km = KMeans(n_clusters=k)
    km = km.fit(dfp)
    ssd.append(km.inertia_)

kneedle = KneeLocator(ks, ssd, S=1.0, curve="convex", direction="decreasing")
kneedle.plot_knee()
plt.show()

k = round(kneedle.knee) # Número óptimo para K
print(f"Número de clusters sugeridos por el método= {k}")

# ======================================
# Ya con el K, calculamos los clusters
# ======================================

kmeans = KMeans(n_clusters=k).fit(dfp)

# Generar el scatterplot con la organización de los clusters
sns.scatterplot(data=df, x="Food%", y="weekday", hue=kmeans.labels_)
plt.show()

# %%
cluster0=df[kmeans.labels_==0]
cluster0[["Food%","Fresh%","Drinks%","Home%","Beauty%","Health%","Baby%","Pets%"]].sum().plot.bar()
plt.show()

# %%
cluster1=df[kmeans.labels_==1]
cluster1[["Food%","Fresh%","Drinks%","Home%","Beauty%","Health%","Baby%","Pets%"]].sum().plot.bar()
plt.show()

# %%
cluster2=df[kmeans.labels_==2]
cluster2[["Food%","Fresh%","Drinks%","Home%","Beauty%","Health%","Baby%","Pets%"]].sum().plot.bar()
plt.show()

# %%
cluster3=df[kmeans.labels_==3]
cluster3[["Food%","Fresh%","Drinks%","Home%","Beauty%","Health%","Baby%","Pets%"]].sum().plot.bar()
plt.show()

# %%
