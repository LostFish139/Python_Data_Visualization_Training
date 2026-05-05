"""
多分类器模块

该模块包含用于多分类任务的机器学习分类器。
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class MultiClassClassifier:
    """
    多分类器类，使用逻辑回归进行多分类

    参数:
        random_state (int): 随机种子
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(max_iter=1000, random_state=random_state)
        self.binary_classifiers = {}

    def fit(self, X, y):
        """
        训练分类器

        参数:
            X (np.array): 特征数据
            y (np.array): 标签数据
        """
        # 数据标准化
        X_std = self.scaler.fit_transform(X)

        # 训练多分类器
        self.classifier.fit(X_std, y)

        # 训练二分类器用于决策边界可视化
        # 分类器1: 类0 vs 类1+2
        y_binary1 = (y != 0).astype(int)
        from sklearn.linear_model import LogisticRegression as BinaryLogisticRegression
        clf1 = BinaryLogisticRegression(max_iter=1000, random_state=self.random_state)
        clf1.fit(X_std, y_binary1)
        self.binary_classifiers['0_vs_12'] = clf1

        # 分类器2: 类1 vs 类2
        mask_12 = (y != 0)
        if np.sum(mask_12) > 0:
            X_12 = X_std[mask_12]
            y_12 = y[mask_12] - 1
            clf2 = BinaryLogisticRegression(max_iter=1000, random_state=self.random_state)
            clf2.fit(X_12, y_12)
            self.binary_classifiers['1_vs_2'] = clf2

        return self

    def predict(self, X):
        """
        预测类别

        参数:
            X (np.array): 特征数据

        返回:
            np.array: 预测的类别
        """
        X_std = self.scaler.transform(X)
        return self.classifier.predict(X_std)

    def predict_proba(self, X):
        """
        预测概率

        参数:
            X (np.array): 特征数据

        返回:
            np.array: 预测的概率
        """
        X_std = self.scaler.transform(X)
        return self.classifier.predict_proba(X_std)

    def score(self, X, y):
        """
        计算准确率

        参数:
            X (np.array): 特征数据
            y (np.array): 真实标签

        返回:
            float: 准确率
        """
        X_std = self.scaler.transform(X)
        return self.classifier.score(X_std, y)

    def get_decision_boundary_params(self, boundary_type='0_vs_12'):
        """
        获取决策边界参数

        参数:
            boundary_type (str): 决策边界类型，可选 '0_vs_12' 或 '1_vs_2'

        返回:
            tuple: (权重, 偏置)
        """
        if boundary_type in self.binary_classifiers:
            clf = self.binary_classifiers[boundary_type]
            return clf.coef_[0], clf.intercept_[0]
        else:
            raise ValueError(f"不支持的决策边界类型: {boundary_type}")
