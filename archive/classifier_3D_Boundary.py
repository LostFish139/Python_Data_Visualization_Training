import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data[:, [0, 2, 3]]
y = (iris.target != 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_std, y_train)

X_std = scaler.transform(X)
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

w, b = clf.coef_[0], clf.intercept_[0]
x1_rng = np.linspace(-3, 3, 50)
x2_rng = np.linspace(-3, 3, 50)
X1, X2 = np.meshgrid(x1_rng, x2_rng)
X3 = -(w[0]*X1 + w[1]*X2 + b) / w[2]
ax.plot_surface(X1, X2, X3, color='gray', alpha=0.4, edgecolor='none')

colors = ['tab:blue', 'tab:orange']
for k in (0, 1):
    mask = y == k
    ax.scatter(X_std[mask, 0],
               X_std[mask, 1],
               X_std[mask, 2],
               c=colors[k], s=50, edgecolor='k', label=f'class {k}')

ax.set_xlabel('sepal length (std)')
ax.set_ylabel('petal length (std)')
ax.set_zlabel('petal width (std)')
ax.set_title('3D Decision Boundary – Two Classes (All 150 Samples)')
ax.legend()
plt.tight_layout()
plt.show()