# 鸢尾花数据分类及可视化项目

本项目使用机器学习算法对经典的鸢尾花数据集进行分类，并提供多种可视化方式来展示分类结果和数据特征。

## 项目结构

```
.
├── README.md                 # 项目说明文档
├── requirements.txt          # 项目依赖
├── .gitignore               # Git忽略文件配置
├── src/                     # 源代码目录
│   ├── classifiers/         # 分类器模块
│   │   ├── __init__.py
│   │   ├── binary_classifier.py    # 二分类器
│   │   └── multi_class_classifier.py  # 多分类器
│   └── utils/               # 工具函数模块
│       ├── __init__.py
│       └── data_processing.py  # 数据处理工具
└── visualizations/          # 可视化脚本目录
    ├── probability_surface.py    # 3D概率曲面可视化
    ├── decision_boundary.py      # 3D决策边界可视化
    └── data_preview.py           # 数据预览可视化
```

## 安装依赖

在运行项目之前，请确保已安装所有必要的依赖包：

```bash
pip install -r requirements.txt
```

## 使用说明

### 1. 数据预览

运行以下命令查看鸢尾花数据集的基本信息和可视化：

```bash
python visualizations/data_preview.py
```

### 2. 3D概率曲面可视化

使用高斯朴素贝叶斯分类器进行二分类，并绘制3D概率曲面：

```bash
python visualizations/probability_surface.py
```

### 3. 3D决策边界可视化

使用逻辑回归进行多分类，并绘制3D决策边界：

```bash
python visualizations/decision_boundary.py
```

## 数据集说明

本项目使用的是经典的鸢尾花数据集（Iris Dataset），包含150个样本，分为3个类别（Setosa、Versicolor和Virginica），每个样本有4个特征：
- 萼片长度 (sepal length)
- 萼片宽度 (sepal width)
- 花瓣长度 (petal length)
- 花瓣宽度 (petal width)

## 机器学习算法

项目使用了以下机器学习算法：
- 高斯朴素贝叶斯 (Gaussian Naive Bayes)
- 逻辑回归 (Logistic Regression)

## 可视化技术

项目使用了多种可视化技术：
- Matplotlib 3D绘图
- Seaborn统计图表
- Plotly交互式图表

## 贡献指南

欢迎提交问题和拉取请求来改进这个项目。

## 许可证

本项目采用MIT许可证。