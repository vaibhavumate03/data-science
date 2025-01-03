# data-science
#clustering using random data

import sklearn 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    "customerid" : range(1,201),
    "annualincome" : np.random.randint(15,150,200),
    "spendingscore" : np.random.randint(1,101,200)
}

print (data)
df = pd.DataFrame(data)

x= df[["annualincome","spendingscore"]]
kmeans = KMeans(n_clusters = 5 , random_state = 42)
kmeans.fit(x)

df["Cluster"]=kmeans.labels_

plt.figure(figsize = (10,6))
for cluster in range(5):
    cluster_data = x [ df["Cluster"]==cluster]
    plt.scatter(cluster_data["annualincome"],cluster_data["spendingscore"],label=f"Cluster {cluster}")

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x', label='centroids')

plt.title("cutomer segmentation")
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.legend()
plt.grid()
plt.show()
