"""
数据预览可视化

该脚本对鸢尾花数据集进行数据预览和可视化，
包括箱线图和交互式散点图。
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sys
import os

# 添加父目录到路径，以便导入项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_iris_data, get_feature_names, get_target_names


def plot_boxplots():
    """
    绘制各特征的箱线图
    """
    # 加载数据
    X, y = load_iris_data()
    feature_names = get_feature_names()
    target_names = get_target_names()

    # 创建DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[i] for i in y]

    # 创建多个子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 第一个子图：sepal_length 的箱线图
    sns.boxplot(x='species', y=feature_names[0], data=df, ax=axes[0, 0])
    axes[0, 0].set_title(f'{feature_names[0]} by Species')

    # 第二个子图：sepal_width 的箱线图
    sns.boxplot(x='species', y=feature_names[1], data=df, ax=axes[0, 1])
    axes[0, 1].set_title(f'{feature_names[1]} by Species')

    # 第三个子图：petal_length 的箱线图
    sns.boxplot(x='species', y=feature_names[2], data=df, ax=axes[1, 0])
    axes[1, 0].set_title(f'{feature_names[2]} by Species')

    # 第四个子图：petal_width 的箱线图
    sns.boxplot(x='species', y=feature_names[3], data=df, ax=axes[1, 1])
    axes[1, 1].set_title(f'{feature_names[3]} by Species')

    # 调整布局，使得子图之间不重叠
    plt.tight_layout()

    # 保存图像
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'iris_boxplots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"箱线图已保存至: {output_path}")

    plt.show()


def plot_scatter_plots():
    """
    绘制特征之间的交互式散点图
    """
    # 加载数据
    X, y = load_iris_data()
    feature_names = get_feature_names()
    target_names = get_target_names()

    # 创建DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[i] for i in y]

    # 创建所有可能的特征组合
    feature_pairs = [
        (feature_names[0], feature_names[1]),
        (feature_names[0], feature_names[2]),
        (feature_names[0], feature_names[3]),
        (feature_names[1], feature_names[2]),
        (feature_names[1], feature_names[3]),
        (feature_names[2], feature_names[3])
    ]

    # 为每个特征对创建散点图
    figures = []
    for x_feat, y_feat in feature_pairs:
        title = f"{x_feat} vs {y_feat}"
        fig = px.scatter(df, x=x_feat, y=y_feat, color='species', title=title)
        figures.append(fig)

    # 显示交互式图表
    for i, fig in enumerate(figures, 1):
        print(f"显示散点图 {i}/{len(figures)}: {feature_pairs[i-1][0]} vs {feature_pairs[i-1][1]}")
        fig.show()

    # 保存交互式图表
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    for i, (fig, (x_feat, y_feat)) in enumerate(zip(figures, feature_pairs), 1):
        filename = f"scatter_{x_feat.replace(' ', '_')}_vs_{y_feat.replace(' ', '_')}.html"
        output_path = os.path.join(output_dir, filename)
        fig.write_html(output_path)
        print(f"散点图已保存至: {output_path}")


def show_data_preview():
    """
    显示数据预览
    """
    # 加载数据
    X, y = load_iris_data()
    feature_names = get_feature_names()
    target_names = get_target_names()

    # 创建DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[i] for i in y]

    # 显示数据集信息
    print("数据集形状:", df.shape)
    print("
数据集前10行:")
    print(df.head(10))
    print("
数据集统计信息:")
    print(df.describe())
    print("
各类别样本数:")
    print(df['species'].value_counts())


if __name__ == '__main__':
    print("选择要执行的操作:")
    print("1. 显示数据预览")
    print("2. 绘制箱线图")
    print("3. 绘制散点图")
    print("4. 执行所有操作")

    choice = input("请输入选项 (1, 2, 3 或 4): ")

    if choice == '1':
        show_data_preview()
    elif choice == '2':
        plot_boxplots()
    elif choice == '3':
        plot_scatter_plots()
    elif choice == '4':
        show_data_preview()
        plot_boxplots()
        plot_scatter_plots()
    else:
        print("无效的选项，将执行所有操作")
        show_data_preview()
        plot_boxplots()
        plot_scatter_plots()
