import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

#load data
bd = pd.read_json('boid_data.json')
link_meth = 'single'

idx = 0

print('num boids', len(bd.iloc[idx,1]))

Z = linkage(bd.iloc[idx,1], method=link_meth)
dendrogram(Z, color_threshold=100)
plt.axhline(y=200, c='red', linestyle='--')

# plot the top three levels of the dendrogram
plt.xlabel("Boid index")
plt.ylabel("Distance")
plt.show()


