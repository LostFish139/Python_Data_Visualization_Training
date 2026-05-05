import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

iris = load_iris()
X = iris.data[:, [0, 2, 3]]  # sepal length, petal length, petal width
y = iris.target

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 分类器1: 类0 vs 类1+2
y_binary1 = (y != 0).astype(int)  # 类0为0，类1和2为1
clf1 = LogisticRegression(max_iter=1000)
clf1.fit(X_std, y_binary1)

# 分类器2: 类1 vs 类2
mask_12 = (y != 0)
X_12 = X_std[mask_12]
y_12 = y[mask_12] - 1
clf2 = LogisticRegression(max_iter=1000)
clf2.fit(X_12, y_12)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制决策面1: 分离类0和类1+2
w1, b1 = clf1.coef_[0], clf1.intercept_[0]
x1_rng = np.linspace(-3, 3, 50)
x2_rng = np.linspace(-3, 3, 50)
X1, X2 = np.meshgrid(x1_rng, x2_rng)
X3_1 = -(w1[0]*X1 + w1[1]*X2 + b1) / w1[2]
ax.plot_surface(X1, X2, X3_1, color='red', alpha=0.3, edgecolor='none')

# 绘制决策面2: 分离类1和类2 (仅在类1和2区域)
w2, b2 = clf2.coef_[0], clf2.intercept_[0]

x1_12_range = X_std[mask_12, 0]
x2_12_range = X_std[mask_12, 1]
x1_min, x1_max = x1_12_range.min(), x1_12_range.max()
x2_min, x2_max = x2_12_range.min(), x2_12_range.max()

x1_12 = np.linspace(x1_min-0.5, x1_max+0.5, 50)
x2_12 = np.linspace(x2_min-0.5, x2_max+0.5, 50)
X1_12, X2_12 = np.meshgrid(x1_12, x2_12)
X3_2 = -(w2[0]*X1_12 + w2[1]*X2_12 + b2) / w2[2]
ax.plot_surface(X1_12, X2_12, X3_2, color='blue', alpha=0.3, edgecolor='none')

colors = ['tab:blue', 'tab:orange', 'tab:green']
class_names = ['Setosa', 'Versicolor', 'Virginica']
for k in range(3):
    mask = y == k
    ax.scatter(X_std[mask, 0],
               X_std[mask, 1],
               X_std[mask, 2],
               c=colors[k], s=60, edgecolor='k', label=class_names[k])

ax.set_xlabel('Sepal Length (std)')
ax.set_ylabel('Petal Length (std)')
ax.set_zlabel('Petal Width (std)')
ax.set_title('3D Decision Boundaries for 3-Class Classification\nTwo Planes Separating Three Classes')
ax.legend()

plt.tight_layout()
plt.show()
