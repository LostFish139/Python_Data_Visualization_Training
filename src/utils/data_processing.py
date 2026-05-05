"""
数据处理工具模块

该模块包含数据加载和预处理的工具函数。
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_iris_data(feature_indices=None, binary_classification=False, class_0=0, class_1=1):
    """
    加载鸢尾花数据集

    参数:
        feature_indices (list): 要使用的特征索引，默认为None（使用所有特征）
        binary_classification (bool): 是否进行二分类
        class_0 (int): 二分类时，第一个类的标签
        class_1 (int): 二分类时，第二个类的标签

    返回:
        tuple: (X, y) 特征数据和标签
    """
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 选择特征
    if feature_indices is not None:
        X = X[:, feature_indices]

    # 二分类
    if binary_classification:
        mask = (y == class_0) | (y == class_1)
        X = X[mask]
        y = y[mask]
        y = (y == class_1).astype(int)

    return X, y


def standardize_features(X_train, X_test=None):
    """
    标准化特征数据

    参数:
        X_train (np.array): 训练集特征数据
        X_test (np.array): 测试集特征数据，可选

    返回:
        tuple: (X_train_scaled, X_test_scaled, scaler) 标准化后的数据和标准化器
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler

    return X_train_scaled, None, scaler


def split_data(X, y, test_size=0.3, random_state=42, stratify=None):
    """
    划分训练集和测试集

    参数:
        X (np.array): 特征数据
        y (np.array): 标签数据
        test_size (float): 测试集比例
        random_state (int): 随机种子
        stratify (np.array): 分层依据，默认为y

    返回:
        tuple: (X_train, X_test, y_train, y_test) 划分后的数据
    """
    if stratify is None:
        stratify = y

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def get_feature_names(feature_indices=None):
    """
    获取特征名称

    参数:
        feature_indices (list): 特征索引列表

    返回:
        list: 特征名称列表
    """
    iris = load_iris()
    feature_names = iris.feature_names

    if feature_indices is not None:
        feature_names = [feature_names[i] for i in feature_indices]

    return feature_names


def get_target_names():
    """
    获取目标类别名称

    返回:
        list: 目标类别名称列表
    """
    iris = load_iris()
    return iris.target_names
