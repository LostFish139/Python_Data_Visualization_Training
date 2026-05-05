"""
二分类器模块

该模块包含用于二分类任务的机器学习分类器。
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class BinaryClassifier:
    """
    二分类器类，支持多种机器学习算法

    参数:
        algorithm (str): 使用的算法，可选 'naive_bayes' 或 'logistic_regression'
        random_state (int): 随机种子
    """

    def __init__(self, algorithm='naive_bayes', random_state=42):
        self.algorithm = algorithm
        self.random_state = random_state
        self.scaler = StandardScaler()

        # 根据算法选择分类器
        if algorithm == 'naive_bayes':
            self.classifier = GaussianNB()
        elif algorithm == 'logistic_regression':
            self.classifier = LogisticRegression(max_iter=1000, random_state=random_state)
        else:
            raise ValueError(f"不支持的算法: {algorithm}")

    def fit(self, X, y):
        """
        训练分类器

        参数:
            X (np.array): 特征数据
            y (np.array): 标签数据
        """
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        return self

    def predict(self, X):
        """
        预测类别

        参数:
            X (np.array): 特征数据

        返回:
            np.array: 预测的类别
        """
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X):
        """
        预测概率

        参数:
            X (np.array): 特征数据

        返回:
            np.array: 预测的概率
        """
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)

    def score(self, X, y):
        """
        计算准确率

        参数:
            X (np.array): 特征数据
            y (np.array): 真实标签

        返回:
            float: 准确率
        """
        X_scaled = self.scaler.transform(X)
        return self.classifier.score(X_scaled, y)

    def get_decision_boundary_params(self):
        """
        获取决策边界参数（仅适用于线性分类器）

        返回:
            tuple: (权重, 偏置)
        """
        if self.algorithm == 'logistic_regression':
            return self.classifier.coef_[0], self.classifier.intercept_[0]
        else:
            raise ValueError(f"算法 {self.algorithm} 不支持获取决策边界参数")
