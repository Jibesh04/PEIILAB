# k Nearest Neighbour
import pandas as pd
import numpy as np
import math

def findEuclideanDistance(point, arr):
  eucl = []
  for elem in arr:
    dist = 0
    for i, p in enumerate(point):
      dist = dist + ((elem[i] - p)**2)
    eucl.append(round(math.sqrt(dist), 1))
  return eucl

def classify(point, arr, label, k):
  dist = findEuclideanDistance(point, arr)
  eucl_dict = {key : value for key, value in enumerate(dist)}
  print('Euclidean Distance obtained\n', eucl_dict)
  count = {}
  for elem in sorted(eucl_dict.items(), key = lambda x: x[1]):
    if not k:
      break
    k = k - 1
    i = label[elem[0]]
    if i not in count.keys():
      count[i] = 0
    count[i] = int(count[i]) + 1
  return max(count.items(), key=lambda x:x[1])[0]

def print_dataset(features, target, point, arr, label, plabel):
  arr.append(point)
  df = pd.DataFrame(arr, columns=features)
  label.append(plabel)
  df[target] = label
  print(df)

def testcase1():
  point = [6.0, 7.0]
  print('Test Data Point:', point)
  arr = [[2.0, 3.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [10.0, 10.0]]
  label = ['A', 'A', 'B', 'B', 'A']
  print('Target Classes:', np.unique(np.array(label)))
  plabel = classify(point, arr, label, 3)
  print('Label:', plabel)
  print_dataset(['x1', 'x2'], 'y', point, arr, label, plabel)

def testcase2():
  point = [7.4, 114]
  print('Test Data Point:', point)
  arr = [[8.0, 160], [6.2, 170], [7.2, 168], [8.2, 155]]
  label = ['Action', 'Action', 'Comedy', 'Comedy']
  print('Target Classes:', np.unique(np.array(label)))
  plabel = classify(point, arr, label, 3)
  print('Label:', plabel)
  print_dataset(['IMDb Rating', 'Duration'], 'Genre', point, arr, label, plabel)

if __name__ == "__main__":
  testcase1()
  print('-'*30)
  testcase2()


# OUTPUT

# Test Data Point: [6.0, 7.0]
# Target Classes: ['A' 'B']
# Euclidean Distance obtained
#  {0: 5.7, 1: 4.2, 2: 1.4, 3: 1.4, 4: 5.0}
# Label: B
#      x1    x2  y
# 0   2.0   3.0  A
# 1   3.0   4.0  A
# 2   5.0   6.0  B
# 3   7.0   8.0  B
# 4  10.0  10.0  A
# 5   6.0   7.0  B
# ------------------------------
# Test Data Point: [7.4, 114]
# Target Classes: ['Action' 'Comedy']
# Euclidean Distance obtained
#  {0: 46.0, 1: 56.0, 2: 54.0, 3: 41.0}
# Label: Comedy
#    IMDb Rating  Duration   Genre
# 0          8.0       160  Action
# 1          6.2       170  Action
# 2          7.2       168  Comedy
# 3          8.2       155  Comedy
# 4          7.4       114  Comedy
