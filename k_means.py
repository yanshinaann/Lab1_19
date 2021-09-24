import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.manifold import MDS

filename = "/Users/anna/Documents/Documents/University /4 курс/ИнтСИС/Интеллектуальные системы. МКН/Наборы данных к заданию 1. Кластерный анализ./19_Зарплата/adult.data"

df = pd.read_table(filename, sep=",")


data = df.iloc[:, [0, 2, 4, 10, 11, 12]].values

wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 15), wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of cluster (k)')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=3)


k = kmeans.fit_predict(data)

df['label'] = k

print(df)

cmd = MDS(n_components=2)
trans = cmd.fit_transform(data)


print(trans.shape)

plt.scatter(trans[k == 0, 0], trans[k == 0, 1], s=10, c='red', label='Cluster 1')
plt.scatter(trans[k == 1, 0], trans[k == 1, 1], s=10, c='blue', label='Cluster 2')
plt.scatter(trans[k == 2, 0], trans[k == 2, 1], s=10, c='green', label='Cluster 3')
plt.show()

writer = pd.ExcelWriter('123.xlsx', engine='xlsxwriter')

from statistics import mode
for i, group in df.groupby('label'):
    print('=' * 10)
    print('cluster {}'.format(i))
    print(group.iloc[:, :-5].values)

    val = [np.sum(x) for x in group.iloc[:, :-5].values]
    print(np.mean(val))
    print(np.mean(group['SEX']))
    print(np.mean(group['ALTER']))
    print(mode(group['HERKU']))
    print(mode(group['FB']))
