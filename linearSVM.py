# Linear SVM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

X_pos = np.array([[3, 1], [3, -1], [6, 1], [6, -1]])
y_pos = np.ones(len(X_pos))
X_neg = np.array([[1, 0], [0, 1], [0, -1], [-1, 0]])
y_neg = -np.ones(len(X_neg))
X = np.concatenate((X_pos, X_neg))
y = np.concatenate((y_pos, y_neg))

svm = SVC(kernel='linear')
svm.fit(X, y)
support_vectors = svm.support_vectors_
w = svm.coef_[0]
bias = svm.intercept_[0]
w = [round(coeff) for coeff in w]
bias = round(bias)
print("Weight:",w)
print("Bias",bias)
plt.scatter(X_pos[:, 0], X_pos[:, 1], marker='*', label='Positive')
plt.scatter(X_neg[:, 0], X_neg[:, 1], marker='s', label='Negative')
if w[1] != 0:
    x_values = np.linspace(-1, 7, 10)
    y_values = np.zeros_like(x_values) + (-bias / w[1])
    plt.plot(x_values, y_values, 'k-', label='Hyperplane')
else:
    x_values = np.full(10, -bias / w[0])  # For vertical line
    y_values = np.linspace(-1, 2, 10)
    plt.plot(x_values, y_values, 'k-', label='Hyperplane')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker='.', label='Support Vectors')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('SVM Classifier')
plt.legend()
plt.grid(True)
plt.show()
