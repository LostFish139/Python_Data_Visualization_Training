import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# 加载Iris数据集（前三个特征，二分类：setosa和versicolor）
iris = load_iris()
X = iris.data[:, :3]  # 萼片长度、萼片宽度、花瓣长度
y = (iris.target < 2).astype(int)  # 二分类：0=setosa, 1=versicolor

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# 训练高斯朴素贝叶斯分类器
clf = GaussianNB()
clf.fit(X_train, y_train)

# 创建3D网格
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
z_min, z_max = X_scaled[:, 2].min() - 0.5, X_scaled[:, 2].max() + 0.5

x_grid = np.linspace(x_min, x_max, 50)
y_grid = np.linspace(y_min, y_max, 50)
z_grid = np.linspace(z_min, z_max, 50)
xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)

# 预测概率
grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
probs = clf.predict_proba(grid_points)[:, 1].reshape(xx.shape)

# 创建图形
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
fig.suptitle('Iris Dataset - Gaussian Naive Bayes 3D Probability Surface', fontsize=16)

# 计算各坐标面的固定值
x_fixed = np.mean(X_scaled[:, 0])
y_fixed = np.mean(X_scaled[:, 1])
z_fixed = np.mean(X_scaled[:, 2])

# 1. 3D空间概率曲面
z_slice_idx = probs.shape[2] // 2
surf_z = probs[:, :, z_slice_idx]
surf = ax.plot_surface(xx[:, :, z_slice_idx],yy[:, :, z_slice_idx],surf_z,
    cmap='coolwarm',alpha=0.8,linewidth=0,ntialiased=True
)

# 2. XY平面投影（固定Z值）
x_grid_2d, y_grid_2d = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)
X_grid_2d = np.c_[x_grid_2d.ravel(), y_grid_2d.ravel(), np.full(x_grid_2d.size, z_fixed)]
probs_xy = clf.predict_proba(X_grid_2d)[:, 1].reshape(x_grid_2d.shape)

# 在XY平面上绘制概率分布
xy_min_z = z_min - 0.3  # 将XY投影面稍微下移
ax.contourf(x_grid_2d, y_grid_2d, probs_xy,
           zdir='z', offset=xy_min_z, cmap='coolwarm', alpha=0.7)

# 3. XZ平面投影（固定Y值）
x_grid_2d, z_grid_2d = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(z_min, z_max, 100)
)
X_grid_2d = np.c_[x_grid_2d.ravel(), np.full(x_grid_2d.size, y_fixed), z_grid_2d.ravel()]
probs_xz = clf.predict_proba(X_grid_2d)[:, 1].reshape(x_grid_2d.shape)

# 在XZ平面上绘制概率分布
xz_max_y = y_max + 0.3  # 将XZ投影面稍微右移
ax.contourf(x_grid_2d, probs_xz, z_grid_2d,
           zdir='y', offset=xz_max_y, cmap='coolwarm', alpha=0.7)

# 4. YZ平面投影（固定X值）
y_grid_2d, z_grid_2d = np.meshgrid(
    np.linspace(y_min, y_max, 100),
    np.linspace(z_min, z_max, 100)
)
X_grid_2d = np.c_[np.full(y_grid_2d.size, x_fixed), y_grid_2d.ravel(), z_grid_2d.ravel()]
probs_yz = clf.predict_proba(X_grid_2d)[:, 1].reshape(y_grid_2d.shape)

# 在YZ平面上绘制概率分布
yz_min_x = x_min - 0.3  # 将YZ投影面稍微左移
ax.contourf(probs_yz, y_grid_2d, z_grid_2d,
           zdir='x', offset=yz_min_x, cmap='coolwarm', alpha=0.7)

# 设置坐标轴标签
ax.set_xlabel('Sepal Length', fontsize=12)
ax.set_ylabel('Sepal Width', fontsize=12)
ax.set_zlabel('Petal Length', fontsize=12)

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)

# 调整视角
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('iris_gaussian_nb_3d_probability_surface.png', dpi=300, bbox_inches='tight')
plt.show()