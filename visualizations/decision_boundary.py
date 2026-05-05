"""
3D决策边界可视化

该脚本使用逻辑回归对鸢尾花数据集进行分类，
并绘制3D决策边界。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# 添加父目录到路径，以便导入项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.classifiers import BinaryClassifier, MultiClassClassifier
from src.utils import load_iris_data, get_feature_names, get_target_names


def plot_binary_decision_boundary():
    """
    绘制二分类3D决策边界
    """
    # 加载数据（前三个特征，二分类：setosa和非setosa）
    X, y = load_iris_data(feature_indices=[0, 2, 3], binary_classification=True, class_0=0, class_1=1)
    feature_names = get_feature_names(feature_indices=[0, 2, 3])

    # 训练分类器
    clf = BinaryClassifier(algorithm='logistic_regression')
    clf.fit(X, y)

    # 获取标准化后的数据
    X_std = clf.scaler.transform(X)

    # 创建图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 获取决策边界参数
    w, b = clf.get_decision_boundary_params()

    # 创建网格
    x1_rng = np.linspace(-3, 3, 50)
    x2_rng = np.linspace(-3, 3, 50)
    X1, X2 = np.meshgrid(x1_rng, x2_rng)
    X3 = -(w[0]*X1 + w[1]*X2 + b) / w[2]

    # 绘制决策边界
    ax.plot_surface(X1, X2, X3, color='gray', alpha=0.4, edgecolor='none')

    # 绘制数据点
    colors = ['tab:blue', 'tab:orange']
    for k in (0, 1):
        mask = y == k
        ax.scatter(X_std[mask, 0],
                   X_std[mask, 1],
                   X_std[mask, 2],
                   c=colors[k], s=50, edgecolor='k', label=f'class {k}')

    # 设置坐标轴标签
    ax.set_xlabel(f'{feature_names[0]} (std)')
    ax.set_ylabel(f'{feature_names[1]} (std)')
    ax.set_zlabel(f'{feature_names[2]} (std)')
    ax.set_title('3D Decision Boundary – Two Classes (All 150 Samples)')
    ax.legend()

    plt.tight_layout()

    # 保存图像
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'binary_decision_boundary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存至: {output_path}")

    plt.show()


def plot_multi_class_decision_boundary():
    """
    绘制多分类3D决策边界
    """
    # 加载数据（前三个特征，三分类）
    X, y = load_iris_data(feature_indices=[0, 2, 3])
    feature_names = get_feature_names(feature_indices=[0, 2, 3])
    target_names = get_target_names()

    # 训练分类器
    clf = MultiClassClassifier()
    clf.fit(X, y)

    # 获取标准化后的数据
    X_std = clf.scaler.transform(X)

    # 创建图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制决策面1: 分离类0和类1+2
    w1, b1 = clf.get_decision_boundary_params('0_vs_12')
    x1_rng = np.linspace(-3, 3, 50)
    x2_rng = np.linspace(-3, 3, 50)
    X1, X2 = np.meshgrid(x1_rng, x2_rng)
    X3_1 = -(w1[0]*X1 + w1[1]*X2 + b1) / w1[2]
    ax.plot_surface(X1, X2, X3_1, color='red', alpha=0.3, edgecolor='none')

    # 绘制决策面2: 分离类1和类2 (仅在类1和2区域)
    w2, b2 = clf.get_decision_boundary_params('1_vs_2')

    mask_12 = (y != 0)
    x1_12_range = X_std[mask_12, 0]
    x2_12_range = X_std[mask_12, 1]
    x1_min, x1_max = x1_12_range.min(), x1_12_range.max()
    x2_min, x2_max = x2_12_range.min(), x2_12_range.max()

    x1_12 = np.linspace(x1_min-0.5, x1_max+0.5, 50)
    x2_12 = np.linspace(x2_min-0.5, x2_max+0.5, 50)
    X1_12, X2_12 = np.meshgrid(x1_12, x2_12)
    X3_2 = -(w2[0]*X1_12 + w2[1]*X2_12 + b2) / w2[2]
    ax.plot_surface(X1_12, X2_12, X3_2, color='blue', alpha=0.3, edgecolor='none')

    # 绘制数据点
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for k in range(3):
        mask = y == k
        ax.scatter(X_std[mask, 0],
                   X_std[mask, 1],
                   X_std[mask, 2],
                   c=colors[k], s=60, edgecolor='k', label=target_names[k])

    # 设置坐标轴标签
    ax.set_xlabel(f'{feature_names[0]} (std)')
    ax.set_ylabel(f'{feature_names[1]} (std)')
    ax.set_zlabel(f'{feature_names[2]} (std)')
    ax.set_title('3D Decision Boundaries for 3-Class Classification\nTwo Planes Separating Three Classes')
    ax.legend()

    plt.tight_layout()

    # 保存图像
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'multi_class_decision_boundary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存至: {output_path}")

    plt.show()


if __name__ == '__main__':
    print("选择要绘制的决策边界类型:")
    print("1. 二分类决策边界")
    print("2. 多分类决策边界")

    choice = input("请输入选项 (1 或 2): ")

    if choice == '1':
        plot_binary_decision_boundary()
    elif choice == '2':
        plot_multi_class_decision_boundary()
    else:
        print("无效的选项，将绘制多分类决策边界")
        plot_multi_class_decision_boundary()
