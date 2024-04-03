import numpy as np
import pandas as pd

def eucl_dist(p1, p2):
  return np.sqrt(np.sum((p1 - p2)**2))

FLOAT_MAX = np.finfo(np.float64).max

def update_centroids(cluster_centers, df):
  for i in range(len(cluster_centers)):
    x = np.mean(df['x'][df['Cluster'] == i + 1])
    y = np.mean(df['y'][df['Cluster'] == i + 1])
    cluster_centers['x'].iloc[i] = x
    cluster_centers['y'].iloc[i] = y

df = pd.DataFrame({'x': [2, 2, 8, 5, 7, 6, 1, 4], 'y': [10, 5, 4, 8, 5, 4, 2, 9]})
print(df)

pd.options.mode.chained_assignment = None

def k_means(k, df):
  tdf = df.copy()
  cluster_centers = tdf.iloc[:k]
  # print(cluster_centers)
  # print(cluster)
  pr_cl = np.ones(k)
  cl = np.zeros(k)
  pr_cr = tdf.iloc[-k: -1]
  # print(pr_cr)
  while True:
    tdf['Cluster'] = np.ones(len(tdf))
    for i in range(len(tdf)):
      point = tdf.iloc[i]
      min_dist = FLOAT_MAX
      cluster = 1
      for j in range(len(cluster_centers)):
        centroid = cluster_centers.iloc[j]
        dist = eucl_dist(point, centroid)
        # print(dist, min_dist)
        if dist < min_dist:
          min_dist = dist
          cluster = j + 1
      tdf['Cluster'].iloc[i] = cluster
    if np.all(pr_cl == cl):
      break
    # print(pr_cr, cluster_centers)
    if np.all(pr_cr.equals(cluster_centers.reset_index(drop=True))):
      break
    pr_cr = cluster_centers
    update_centroids(cluster_centers, tdf)
    pr_cl = cl
    cl = tdf['Cluster'].unique()
  print(tdf)
  print(cluster_centers)
  return tdf, cluster_centers
print(df)
cdf, centroids = k_means(3, df)
