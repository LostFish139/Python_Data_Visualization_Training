# 项目结构说明

本文档详细说明了重构后的项目结构及其各部分的功能。

## 目录结构

```
.
├── README.md                 # 项目说明文档
├── requirements.txt          # 项目依赖
├── .gitignore               # Git忽略文件配置
├── main.py                  # 项目主入口文件
├── src/                     # 源代码目录
│   ├── classifiers/         # 分类器模块
│   │   ├── __init__.py
│   │   ├── binary_classifier.py    # 二分类器实现
│   │   └── multi_class_classifier.py  # 多分类器实现
│   └── utils/               # 工具函数模块
│       ├── __init__.py
│       └── data_processing.py  # 数据处理工具
├── visualizations/          # 可视化脚本目录
│   ├── probability_surface.py    # 3D概率曲面可视化
│   ├── decision_boundary.py      # 3D决策边界可视化
│   ├── data_preview.py           # 数据预览可视化
│   └── outputs/                  # 可视化输出目录
└── PROJECT_STRUCTURE.md     # 本文件，项目结构说明
```

## 各部分功能说明

### 1. 主入口文件 (main.py)

`main.py` 是项目的主入口文件，提供了统一的用户界面，用户可以通过命令行选择要执行的功能：
- 数据预览
- 3D概率曲面可视化
- 3D决策边界可视化

### 2. 源代码目录 (src/)

#### 2.1 分类器模块 (src/classifiers/)

该模块包含用于鸢尾花数据集分类的各种机器学习分类器：

- `binary_classifier.py`: 实现了二分类器，支持高斯朴素贝叶斯和逻辑回归算法
- `multi_class_classifier.py`: 实现了多分类器，使用逻辑回归进行多分类

#### 2.2 工具函数模块 (src/utils/)

该模块包含数据处理和可视化相关的工具函数：

- `data_processing.py`: 提供了数据加载、预处理和分割的函数

### 3. 可视化脚本目录 (visualizations/)

该目录包含各种可视化脚本：

- `probability_surface.py`: 使用高斯朴素贝叶斯分类器进行二分类，并绘制3D概率曲面
- `decision_boundary.py`: 使用逻辑回归进行分类，并绘制3D决策边界
- `data_preview.py`: 对鸢尾花数据集进行数据预览和可视化，包括箱线图和交互式散点图

### 4. 配置文件

- `requirements.txt`: 列出了项目所需的所有Python依赖包
- `.gitignore`: 指定了Git版本控制时应忽略的文件和目录
- `README.md`: 项目的主要说明文档，包含安装、使用和贡献指南

## 使用方法

### 通过主入口文件运行

```bash
python main.py
```

然后按照提示选择要执行的功能。

### 直接运行可视化脚本

```bash
# 数据预览
python visualizations/data_preview.py

# 3D概率曲面可视化
python visualizations/probability_surface.py

# 3D决策边界可视化
python visualizations/decision_boundary.py
```

### 使用模块中的类和函数

```python
from src.classifiers import BinaryClassifier, MultiClassClassifier
from src.utils import load_iris_data, standardize_features

# 加载数据
X, y = load_iris_data(binary_classification=True)

# 创建并训练分类器
clf = BinaryClassifier(algorithm='naive_bayes')
clf.fit(X, y)

# 预测
predictions = clf.predict(X)
```

## 输出文件

所有可视化结果将保存在 `visualizations/outputs/` 目录中，包括：
- PNG格式的静态图像
- HTML格式的交互式图表

## 扩展项目

要扩展项目功能，可以：

1. 在 `src/classifiers/` 中添加新的分类器
2. 在 `src/utils/` 中添加新的工具函数
3. 在 `visualizations/` 中添加新的可视化脚本
4. 在 `main.py` 中添加新的菜单选项

## 代码规范

项目遵循以下代码规范：
- 使用PEP 8风格指南
- 添加详细的文档字符串
- 使用有意义的变量和函数名
- 保持函数和类的单一职责原则
