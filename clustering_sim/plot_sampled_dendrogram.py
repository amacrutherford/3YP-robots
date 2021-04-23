import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

#load data
bd = pd.read_json('boid_data.json')

idx = 0
link_meth = 'single'

plt.figure(1)
bdsub = random.sample(bd.iloc[idx,1], 10)
print(bdsub)
plt.scatter(*zip(*bdsub))
plt.xlabel('X (m)')
plt.ylabel('Y (m)')

print('num boids', len(bdsub))
plt.figure(2)
Z = linkage(bdsub, method=link_meth)
dendrogram(Z, color_threshold=100)
plt.axhline(y=200, c='red', linestyle='--')

# plot the top three levels of the dendrogram
plt.xlabel("Index")
plt.ylabel("Distance")
plt.show()


