# %%
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator

df = pd.read_csv("Mall_Customers.csv")


# ================================
# Determina el número K óptimo
# ================================
dfp = df[["Annual_Income_(k$)", "Spending_Score"]]

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
print(f"Number of clusters suggested by knee method: {k}")

# ================================
# Ya con el K, calculamos los clusters
# ================================
kmeans = KMeans(n_clusters=k).fit(df[["Annual_Income_(k$)", "Spending_Score"]])

# Generar el scatterplot con la organización de los clusters
sns.scatterplot(data=df, x="Annual_Income_(k$)", y="Spending_Score", hue=kmeans.labels_)
plt.show()

# %%
sns.scatterplot(data=df, x="Annual_Income_(k$)", y="Spending_Score", hue="Age")


# %%
cluster0 = df[kmeans.labels_ == 0]
cluster0.describe()

# %%
cluster1 = df[kmeans.labels_ == 1]
cluster1.describe()

