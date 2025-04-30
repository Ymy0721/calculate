import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from ..utils.gpu_utils import get_array_module

def entropy_weight(data, use_gpu=False):
    """
    使用熵权法计算指标权重，支持GPU加速
    
    参数:
    - data: 指标数据矩阵
    - use_gpu: 是否使用GPU加速
    
    返回:
    - 各指标权重
    """
    epsilon = 1e-12  # 避免除零错误和对数计算错误的小值
    
    # 检查是否使用GPU
    if use_gpu:
        xp = get_array_module()
        data = xp.array(data)
    else:
        xp = np

    # 计算每列的和
    col_sum = xp.sum(data, axis=0)
    # 归一化处理
    p = data / (col_sum + epsilon)
    
    # 计算熵值
    log_p = xp.log(p + epsilon)
    entropy = -xp.sum(p * log_p, axis=0) / xp.log(len(data) + epsilon)
    
    # 计算差异系数
    diff_coef = 1 - entropy
    
    # 计算权重
    if xp.sum(diff_coef) == 0:
        weights = xp.ones_like(diff_coef) / len(diff_coef)
    else:
        weights = diff_coef / xp.sum(diff_coef)
    
    # 如果是GPU数据，转回CPU
    if use_gpu:
        weights = xp.asnumpy(weights) if hasattr(xp, 'asnumpy') else np.array(weights)
    
    return weights

def normalize_indicators(df, indicator_names):
    """
    归一化指标数据
    
    参数:
    - df: 包含指标的DataFrame
    - indicator_names: 指标列名列表
    
    返回:
    - 归一化后的指标数据
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(df[indicator_names])

def calculate_comprehensive_score(metrics_data, weights):
    """
    计算综合评分
    
    参数:
    - metrics_data: 归一化后的指标数据
    - weights: 各指标权重
    
    返回:
    - 综合评分数组
    """
    return np.dot(metrics_data, weights)
