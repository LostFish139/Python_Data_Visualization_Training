"""
鸢尾花数据分类及可视化项目主入口文件

该文件提供了项目的统一入口，用户可以通过命令行参数选择要执行的功能。
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def print_menu():
    """
    打印主菜单
    """
    print("
" + "="*50)
    print("鸢尾花数据分类及可视化项目")
    print("="*50)
    print("请选择要执行的功能:")
    print("1. 数据预览")
    print("2. 3D概率曲面可视化")
    print("3. 3D决策边界可视化")
    print("4. 退出")
    print("="*50)


def run_data_preview():
    """
    运行数据预览功能
    """
    from visualizations.data_preview import show_data_preview, plot_boxplots, plot_scatter_plots

    print("
数据预览选项:")
    print("1. 显示数据预览")
    print("2. 绘制箱线图")
    print("3. 绘制散点图")
    print("4. 执行所有操作")
    print("0. 返回主菜单")

    choice = input("请输入选项: ")

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
    elif choice == '0':
        return
    else:
        print("无效的选项，将执行所有操作")
        show_data_preview()
        plot_boxplots()
        plot_scatter_plots()


def run_probability_surface():
    """
    运行3D概率曲面可视化功能
    """
    from visualizations.probability_surface import plot_3d_probability_surface
    plot_3d_probability_surface()


def run_decision_boundary():
    """
    运行3D决策边界可视化功能
    """
    from visualizations.decision_boundary import plot_binary_decision_boundary, plot_multi_class_decision_boundary

    print("
决策边界可视化选项:")
    print("1. 二分类决策边界")
    print("2. 多分类决策边界")
    print("0. 返回主菜单")

    choice = input("请输入选项: ")

    if choice == '1':
        plot_binary_decision_boundary()
    elif choice == '2':
        plot_multi_class_decision_boundary()
    elif choice == '0':
        return
    else:
        print("无效的选项，将绘制多分类决策边界")
        plot_multi_class_decision_boundary()


def main():
    """
    主函数
    """
    while True:
        print_menu()
        choice = input("请输入选项 (1-4): ")

        if choice == '1':
            run_data_preview()
        elif choice == '2':
            run_probability_surface()
        elif choice == '3':
            run_decision_boundary()
        elif choice == '4':
            print("感谢使用鸢尾花数据分类及可视化项目，再见！")
            sys.exit(0)
        else:
            print("无效的选项，请重新输入")


if __name__ == '__main__':
    main()
