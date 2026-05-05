"""
3D概率曲面可视化

该脚本使用高斯朴素贝叶斯分类器对鸢尾花数据集进行二分类，
并绘制3D概率曲面和其在各个坐标面上的投影。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# 添加父目录到路径，以便导入项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.classifiers import BinaryClassifier
from src.utils import load_iris_data, get_feature_names


def plot_3d_probability_surface():
    """
    绘制3D概率曲面及其投影
    """
    # 加载数据（前三个特征，二分类：setosa和versicolor）
    X, y = load_iris_data(feature_indices=[0, 1, 2], binary_classification=True, class_0=0, class_1=1)
    feature_names = get_feature_names(feature_indices=[0, 1, 2])

    # 训练分类器
    clf = BinaryClassifier(algorithm='naive_bayes')
    clf.fit(X, y)

    # 获取标准化后的数据
    X_scaled = clf.scaler.transform(X)

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

    # 绘制3D概率曲面
    surf = ax.plot_surface(
        xx[:, :, z_slice_idx],
        yy[:, :, z_slice_idx],
        surf_z,
        cmap='coolwarm',
        alpha=0.8,
        linewidth=0,
        antialiased=True
    )

    # 2. XY平面投影（固定Z值）
    x_grid_2d, y_grid_2d = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    X_grid_2d = np.c_[x_grid_2d.ravel(), y_grid_2d.ravel(), np.full(x_grid_2d.size, z_fixed)]
    probs_xy = clf.predict_proba(X_grid_2d)[:, 1].reshape(x_grid_2d.shape)

    # 在XY平面上绘制概率分布
    xy_min_z = z_min - 0.3
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
    xz_max_y = y_max + 0.3
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
    yz_min_x = x_min - 0.3
    ax.contourf(probs_yz, y_grid_2d, z_grid_2d,
           zdir='x', offset=yz_min_x, cmap='coolwarm', alpha=0.7)

    # 设置坐标轴标签
    ax.set_xlabel(feature_names[0], fontsize=12)
    ax.set_ylabel(feature_names[1], fontsize=12)
    ax.set_zlabel(feature_names[2], fontsize=12)

    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)

    # 调整视角
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    # 保存图像
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'iris_gaussian_nb_3d_probability_surface.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存至: {output_path}")

    plt.show()


if __name__ == '__main__':
    plot_3d_probability_surface()
