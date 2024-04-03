import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
eps = np.finfo(float).eps

def transform(points):
  distances = np.sqrt(np.sum(points**2, axis=1))
  for i, distance in enumerate(distances):
    if distance > 2:
        points[i] = [4 - points[i][1] + abs(points[i][0] - points[i][1]),
                     4 - points[i][0] + abs(points[i][0] - points[i][1])]
  return points

X_pos = transform(np.array([[2, 2], [2, -2], [-2, -2], [-2, 2]]))
y_pos = np.ones(len(X_pos))
X_neg = transform(np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]))
y_neg = -np.ones(len(X_neg))
X = np.concatenate((X_pos, X_neg))
y = np.concatenate((y_pos, y_neg))

# print(y)

svm = SVC(kernel='linear')
svm.fit(X, y)
support_vectors = svm.support_vectors_
w_t = (svm.coef_[0])
w=[round(coeff) for coeff in w_t]
bias = round(svm.intercept_[0])

print("Weight:",w)
print("Bias:",bias)

plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], marker='*', label='Positive')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], marker='s', label='Negative')

# Plot hyperplane
if w[1] != 0:
    x_values = np.linspace(-4, 4, 10)
    y_values = (-bias - w[0] * x_values) / (w[1] + eps)
    plt.plot(x_values, y_values, 'k-', label='Hyperplane')
else:
    plt.axvline(x=-bias / (w[0] + eps), color='k', label='Hyperplane')

plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker='.', label='Support Vectors')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('SVM Classifier')
plt.legend()
plt.grid(True)
plt.show()
