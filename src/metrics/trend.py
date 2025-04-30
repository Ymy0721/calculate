import numpy as np
from scipy.stats import linregress

def calculate_trend(entity_set, entity_lib):
    """
    计算趋势指标 - 基于实体的年份分布斜率
    """
    trend_scores = []
    
    for e in entity_set:
        if e in entity_lib:
            years = entity_lib[e]['years']
            if len(years) >= 2:
                # 使用线性回归计算趋势斜率
                try:
                    slope = linregress(range(len(years)), years).slope
                    trend_scores.append(slope)
                except Exception:
                    continue
    
    return np.mean(trend_scores) if trend_scores else 0
