"""
数据处理工具模块 - 提供数据处理和转换的实用函数
"""
import pandas as pd
import numpy as np
import logging
import os
# 导入 scipy.stats 用于计算 zscore
from scipy import stats
import traceback # 确保 traceback 已导入

logger = logging.getLogger(__name__)

def fill_zeros_with_nonzero_mean(df, column_name):
    """
    将指定列中的零值用该列非零值的均值填充
    
    参数:
    - df: pandas DataFrame, 包含要处理的数据
    - column_name: str, 要处理的列名
    
    返回:
    - pandas DataFrame, 处理后的数据框
    """
    # 复制数据框避免修改原始数据
    result_df = df.copy()
    
    # 检查列是否存在
    if (column_name not in result_df.columns):
        logger.warning(f"列 '{column_name}' 不存在于数据框中")
        return result_df
    
    # 检查列中是否有零值
    zero_count = (result_df[column_name] == 0).sum()
    if (zero_count == 0):
        logger.info(f"列 '{column_name}' 中没有零值需要填充")
        return result_df
    
    # 计算非零值的均值
    nonzero_values = result_df[result_df[column_name] != 0][column_name]
    
    # 如果没有非零值，则无法计算均值
    if (len(nonzero_values) == 0):
        logger.warning(f"列 '{column_name}' 中没有非零值，无法计算填充值")
        return result_df
    
    nonzero_mean = nonzero_values.mean()
    logger.info(f"列 '{column_name}' 的非零均值为: {nonzero_mean:.4f}")
    
    # 用非零均值填充零值
    result_df.loc[result_df[column_name] == 0, column_name] = nonzero_mean
    logger.info(f"已将列 '{column_name}' 中的 {zero_count} 个零值填充为非零均值")
    
    return result_df

def normalize_min_max(df, column_name, feature_range=(0, 1)):
    """
    使用Min-Max方法对列进行归一化：X_norm = (X - X_min) / (X_max - X_min)
    
    参数:
    - df: pandas DataFrame, 包含要处理的数据
    - column_name: str, 要处理的列名
    - feature_range: tuple, 归一化的目标区间，默认为(0, 1)
    
    返回:
    - pandas DataFrame, 处理后的数据框
    """
    # 复制数据框避免修改原始数据
    result_df = df.copy()
    
    # 检查列是否存在
    if (column_name not in result_df.columns):
        logger.warning(f"列 '{column_name}' 不存在于数据框中")
        return result_df
    
    # 获取最大值和最小值
    col_min = result_df[column_name].min()
    col_max = result_df[column_name].max()
    
    # 检查是否可以归一化
    if (col_max == col_min):
        logger.warning(f"列 '{column_name}' 的最大值等于最小值 ({col_min})，无法进行Min-Max归一化")
        return result_df
    
    # 提取目标范围
    min_range, max_range = feature_range
    
    # 应用Min-Max归一化公式
    result_df[column_name] = min_range + (result_df[column_name] - col_min) * (max_range - min_range) / (col_max - col_min)
    logger.info(f"列 '{column_name}' 已完成Min-Max归一化，映射到区间[{min_range}, {max_range}]")
    
    return result_df

def normalize_zscore_sigmoid(df, column_name):
    """
    使用 Z-Score 标准化后应用 Sigmoid 函数进行归一化，映射到 (0, 1)
    
    参数:
    - df: pandas DataFrame, 包含要处理的数据
    - column_name: str, 要处理的列名
    
    返回:
    - pandas DataFrame, 处理后的数据框
    """
    # 复制数据框避免修改原始数据
    result_df = df.copy()
    
    # 检查列是否存在
    if column_name not in result_df.columns:
        logger.warning(f"列 '{column_name}' 不存在于数据框中")
        return result_df
        
    # 获取列数据
    col_data = result_df[column_name].astype(float) # 确保是浮点数
    
    # 计算 Z-Score
    # 使用 ddof=0 计算总体标准差，如果样本量小，用 ddof=1 计算样本标准差
    z_scores = stats.zscore(col_data, ddof=0) 
    
    # 处理 Z-Score 计算结果中的 NaN (如果原始数据标准差为0)
    if np.isnan(z_scores).any():
        logger.warning(f"列 '{column_name}' 的标准差为0或包含NaN，Z-Score无法计算，将值设为0.5")
        # 标准差为0意味着所有值相同，Z-Score为NaN。Sigmoid(0)=0.5，所以设为0.5是合理的中间值。
        result_df[column_name] = 0.5 
        return result_df

    # 应用 Sigmoid 函数: S(z) = 1 / (1 + exp(-z))
    sigmoid_values = 1 / (1 + np.exp(-z_scores))
    
    result_df[column_name] = sigmoid_values
    logger.info(f"列 '{column_name}' 已完成 Z-Score + Sigmoid 归一化")
    
    return result_df

def process_metrics_file(file_path, columns=None, save_path=None, normalize_method='minmax'):
    """
    处理指标结果CSV文件，将指定列中的零值用非零均值填充，并进行归一化处理
    
    参数:
    - file_path: str, 指标结果CSV文件路径
    - columns: list or str, 要处理的列名，None表示处理所有数值列
    - save_path: str, 保存结果的路径，None表示不保存
    - normalize_method: str, 归一化方法 ('minmax', 'zscore_sigmoid', 'none')
    
    返回:
    - pandas DataFrame, 处理后的数据框
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        logger.info(f"成功读取文件: {file_path}, 包含 {len(df)} 条记录")
        
        # 如果未指定列，则处理所有数值列
        if (columns is None):
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # 排除ID列等
            columns = [col for col in columns if col not in ['ID', 'Year', 'index']]
            
        # 如果是单个列名，转换为列表
        if isinstance(columns, str):
            columns = [columns]
            
        # 处理每一列
        result_df = df.copy()
        for column in columns:
            if column not in result_df.columns:
                 logger.warning(f"列 '{column}' 在 {file_path} 中不存在，跳过处理")
                 continue
                 
            # 先用非零均值填充零值
            result_df = fill_zeros_with_nonzero_mean(result_df, column)
            
            # 进行归一化处理
            method = normalize_method.lower()
            if method == 'minmax':
                result_df = normalize_min_max(result_df, column)
            elif method == 'zscore_sigmoid':
                result_df = normalize_zscore_sigmoid(result_df, column)
            elif method == 'none':
                logger.info(f"列 '{column}' 选择不进行归一化")
            else:
                logger.warning(f"未知的归一化方法: '{normalize_method}'，跳过列 '{column}' 的归一化")
            
        # 保存结果
        if (save_path):
            # 确保输出目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            result_df.to_csv(save_path, index=False)
            logger.info(f"处理后的结果已保存至: {save_path}")
            
        return result_df
        
    except Exception as e:
        logger.error(f"处理文件 '{file_path}' 时出错: {e}")
        logger.error(traceback.format_exc()) # 打印详细错误堆栈
        raise

def process_and_minmax_normalize(file_path, columns=None, save_path=None, feature_range=(0, 1)):
    """
    处理指标结果CSV文件，先用非零均值填充零值，然后进行Min-Max归一化
    
    参数:
    - file_path: str, 指标结果CSV文件路径
    - columns: list or str, 要处理的列名，None表示处理所有数值列
    - save_path: str, 保存结果的路径，None表示不保存
    - feature_range: tuple, 归一化的目标区间，默认为(0, 1)
    
    返回:
    - pandas DataFrame, 处理后的数据框
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        logger.info(f"成功读取文件: {file_path}, 包含 {len(df)} 条记录")
        
        # 如果未指定列，则处理所有数值列
        if (columns is None):
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # 排除ID列
            columns = [col for col in columns if col not in ['ID']]
            
        # 如果是单个列名，转换为列表
        if isinstance(columns, str):
            columns = [columns]
            
        # 处理每一列
        result_df = df.copy()
        for column in columns:
            # 先用非零均值填充零值
            result_df = fill_zeros_with_nonzero_mean(result_df, column)
            
            # 进行Min-Max归一化
            result_df = normalize_min_max(result_df, column, feature_range)
            
        # 保存结果
        if (save_path):
            # 确保输出目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            result_df.to_csv(save_path, index=False)
            logger.info(f"处理后的结果已保存至: {save_path}")
            
        return result_df
        
    except Exception as e:
        logger.error(f"处理文件时出错: {e}")
        raise

def main():
    """
    示例用法：读取 metrics_results.csv，生成填充零值和不同归一化方法的文件
    """
    # 配置日志记录器
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 定义文件路径
    input_file_path = "data/output/metrics_results.csv"
    save_path_filled = "data/output/metrics_results_filled.csv"
    save_path_minmax = "data/output/metrics_results_minmax.csv"
    save_path_zscore_sigmoid = "data/output/metrics_results_zscore_sigmoid.csv" # 新增保存路径
    
    # 定义要处理的指标列 (包括 Score)
    indicators_and_score = ['Novelty', 'Trend', 'Applicability', 'Dependency', 'Replaceability', 'Maturity', 'Score']
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file_path):
        logger.error(f"输入文件不存在: {input_file_path}")
        return

    try:
        # --- 1. 生成填充零值的文件 (metrics_results_filled.csv) ---
        logger.info(f"\n--- 开始处理以生成填充零值的文件: {save_path_filled} ---")
        df_filled = process_metrics_file(
            input_file_path, 
            columns=indicators_and_score, 
            save_path=save_path_filled,
            normalize_method='none' # 只填充，不归一化
        )
        print(f"\n填充零值结果 ({save_path_filled}):")
        print(df_filled.head())

        # --- 2. 生成填充零值并进行 Min-Max 归一化的文件 (metrics_results_minmax.csv) ---
        logger.info(f"\n--- 开始处理以生成填充零值并 Min-Max 归一化的文件: {save_path_minmax} ---")
        # 注意：process_and_minmax_normalize 内部调用了 fill 和 normalize_min_max
        normalized_minmax_df = process_and_minmax_normalize(
            file_path=input_file_path, # 直接处理原始文件
            columns=indicators_and_score,
            save_path=save_path_minmax
        )
        print(f"\n填充零值并 Min-Max 归一化结果 ({save_path_minmax}):")
        print(normalized_minmax_df.head())

        # --- 3. 生成填充零值并进行 Z-Score + Sigmoid 归一化的文件 ---
        logger.info(f"\n--- 开始处理以生成填充零值并 Z-Score + Sigmoid 归一化的文件: {save_path_zscore_sigmoid} ---")
        normalized_zscore_df = process_metrics_file(
            input_file_path,
            columns=indicators_and_score,
            save_path=save_path_zscore_sigmoid,
            normalize_method='zscore_sigmoid' # 使用新方法
        )
        print(f"\n填充零值并 Z-Score + Sigmoid 归一化结果 ({save_path_zscore_sigmoid}):")
        print(normalized_zscore_df.head())


    except Exception as e:
        logger.error(f"在 main 函数中处理文件时发生错误: {e}")
        logger.error(traceback.format_exc()) # 打印详细错误堆栈
        
if __name__ == "__main__":
    main()