import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import logging
import json # 导入 json 模块

from src.metrics import (
    calculate_novelty, 
    calculate_trend, 
    calculate_applicability,
    calculate_dependency,
    calculate_replaceability,
    calculate_maturity
)
from ..utils.gpu_utils import check_gpu_available, get_array_module
from ..core.weight_calculator import entropy_weight, normalize_indicators, calculate_comprehensive_score

logger = logging.getLogger(__name__)

def calculate_metrics_batch(args):
    """
    批量计算指标，支持GPU加速
    
    参数:
    - args: 元组，包含(文档ID列表, tfidf矩阵, 实体列表, 实体库, 历史tfidf均值, 是否使用GPU, ID到索引的映射)
    """
    batch_doc_ids, tfidf_matrix, entities_list, entity_lib, historical_tfidf_mean, use_gpu, id_to_idx_map = args
    batch_metrics = []

    for doc_id in batch_doc_ids:
        # 获取文档的索引
        idx = id_to_idx_map[doc_id]
        
        # 获取当前文档的实体集合
        ents = entities_list[idx]
        current_set = set(ents) if isinstance(ents, list) else set()
        
        if not current_set:
            # 如果实体集合为空，添加零指标
            metrics_dict = {
                'ID': doc_id,  # 使用原始ID
                'Novelty': 0,
                'Trend': 0,
                'Applicability': 0,
                'Dependency': 0,
                'Replaceability': 0,
                'Maturity': 0,
                'Score': 0  # 添加得分
            }
            batch_metrics.append(metrics_dict)
            continue
            
        try:
            # 计算各指标
            novelty = calculate_novelty(idx, tfidf_matrix, historical_tfidf_mean, use_gpu)
            trend = calculate_trend(current_set, entity_lib)
            applicability = calculate_applicability(current_set, entity_lib, tfidf_matrix, use_gpu)
            dependency = calculate_dependency(current_set, entity_lib)
            replaceability = calculate_replaceability(idx, tfidf_matrix, use_gpu)
            maturity = calculate_maturity(current_set, entity_lib)
            
            # 构建指标字典
            metrics_dict = {
                'ID': doc_id,  # 使用原始ID
                'Novelty': novelty,
                'Trend': trend,
                'Applicability': applicability,
                'Dependency': dependency,
                'Replaceability': replaceability,
                'Maturity': maturity,
                'Score': 0  # 先添加占位，后面再计算真实得分
            }
            
            batch_metrics.append(metrics_dict)
        except Exception as e:
            logger.error(f"计算文档 {doc_id} 指标时出错: {e}")
            # 添加零指标作为fallback
            metrics_dict = {
                'ID': doc_id,  # 使用原始ID
                'Novelty': 0,
                'Trend': 0,
                'Applicability': 0,
                'Dependency': 0,
                'Replaceability': 0,
                'Maturity': 0,
                'Score': 0  # 添加得分
            }
            batch_metrics.append(metrics_dict)

    return batch_metrics

def compute_historical_tfidf_mean(tfidf_matrix):
    """计算历史TF-IDF均值"""
    if hasattr(tfidf_matrix, 'toarray'):
        # 对于大型稀疏矩阵，避免全部转换为密集矩阵
        return np.array(tfidf_matrix.mean(axis=0)).flatten()
    else:
        return np.mean(tfidf_matrix, axis=0)

def load_existing_results(output_file):
    """加载已有计算结果"""
    if os.path.exists(output_file):
        try:
            df_existing = pd.read_csv(output_file)
            processed = set(df_existing['ID'].tolist())
            print(f"发现已有 {len(processed)} 份文档的计算结果。将继续处理剩余文档。")
            return df_existing, processed
        except Exception as ex:
            print(f"读取已有结果出错: {ex}")
            return None, set()
    else:
        print(f"未发现已有结果，将从头开始计算。")
        return None, set()

def save_batch_results(batch_df, output_file, header=False):
    """保存批量计算结果"""
    batch_df.to_csv(output_file, mode='a', index=False, header=header)

def calculate_all_metrics(df, entities_list, entity_lib, output_file='metrics_results.csv', 
                         batch_size=25, num_processes=0, use_gpu=False):
    """
    计算所有文档的所有指标
    
    参数:
    - df: 数据框
    - entities_list: 实体列表
    - entity_lib: 实体库
    - output_file: 输出文件路径
    - batch_size: 批处理大小
    - num_processes: 使用的进程数，0表示自动选择
    - use_gpu: 是否使用GPU加速（已经在main.py中确认过）
    """
    # 设置进程数
    if num_processes <= 0:
        NUM_PROCESSES = max(1, mp.cpu_count() - 1)
    else:
        NUM_PROCESSES = num_processes
        print(f"指定使用 {NUM_PROCESSES} 个进程")
    
    # 准备TF-IDF矩阵
    print("计算TF-IDF矩阵...")
    documents = [' '.join(ents) if isinstance(ents, list) else '' for ents in entities_list]
    tfidf_vectorizer = TfidfVectorizer(min_df=2)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    print(f"TF-IDF矩阵计算完成，形状: {tfidf_matrix.shape}")
    
    # 计算历史TF-IDF均值
    historical_tfidf_mean = compute_historical_tfidf_mean(tfidf_matrix)
    
    # 加载已有结果
    existing_df, processed_ids = load_existing_results(output_file)
    
    # 创建ID到索引的映射
    id_to_idx_map = {id_val: idx for idx, id_val in enumerate(df['ID'])}
    
    # 获取所有ID
    all_ids = df['ID'].tolist()
    
    # 找出未处理的ID
    remaining_ids = [id for id in all_ids if id not in processed_ids]
    print(f"剩余需要处理的文档数: {len(remaining_ids)}")
    
    # 如果所有文档都已处理，直接加载已有结果并重新计算得分
    if not remaining_ids and existing_df is not None:
        print("所有文档都已处理完毕，重新计算综合得分...")
        all_metrics_df = existing_df
    else:
        # 分批处理
        batches = [remaining_ids[i:i + batch_size] for i in range(0, len(remaining_ids), batch_size)]
        
        # 准备多进程参数
        tasks = [(batch, tfidf_matrix, entities_list, entity_lib, historical_tfidf_mean, use_gpu, id_to_idx_map) 
                for batch in batches]
        
        # 存储所有批次的指标结果
        all_batch_results = []
        
        # 多进程计算指标
        with mp.Pool(NUM_PROCESSES) as pool:
            with tqdm(total=len(remaining_ids), desc="计算指标中", unit="doc") as pbar:
                for batch_result in pool.imap_unordered(calculate_metrics_batch, tasks):
                    # 转换为DataFrame
                    batch_df = pd.DataFrame(batch_result)
                    
                    # 保存批次结果到列表中，用于后续处理
                    all_batch_results.append(batch_df)
                    
                    # 更新进度条
                    pbar.update(len(batch_result))
                    pbar.set_postfix(remaining=len(remaining_ids) - pbar.n)
        
        # 合并所有批次结果
        if all_batch_results:
            new_metrics_df = pd.concat(all_batch_results, ignore_index=True)
            
            # 合并新旧结果
            all_metrics_df = pd.concat([existing_df, new_metrics_df]) if existing_df is not None else new_metrics_df
        else:
            all_metrics_df = existing_df
    
    # 计算指标权重和综合得分
    print("计算指标权重和综合得分...")
    indicator_names = ['Novelty', 'Trend', 'Applicability', 'Dependency', 'Replaceability', 'Maturity']
    
    # 检查是否有空值，如果有则填充为0
    nan_count = {col: all_metrics_df[col].isna().sum() for col in indicator_names}
    if sum(nan_count.values()) > 0:
        print(f"指标中存在空值: {nan_count}")
        all_metrics_df = all_metrics_df.fillna({col: 0 for col in indicator_names})
    
    # 获取原始指标数据
    metrics_data = all_metrics_df[indicator_names].values
    
    # 归一化指标
    scaler = normalize_indicators(all_metrics_df, indicator_names)
    
    # 计算权重
    weights = entropy_weight(scaler, use_gpu=use_gpu)
    print("指标权重: " + ", ".join([f"{name}: {w:.3f}" for name, w in zip(indicator_names, weights)]))
    
    # 准备权重字典用于打印和保存
    weights_dict = {name: w for name, w in zip(indicator_names, weights)}
    
    # 打印权重 (确保日志级别允许 INFO)
    weights_log_str = "指标权重: " + ", ".join([f"{name}: {w:.4f}" for name, w in weights_dict.items()])
    print(weights_log_str) # 保留控制台打印
    logger.info(weights_log_str) # 添加日志记录
    
    # 保存权重到 JSON 文件
    output_dir = os.path.dirname(output_file)
    weights_file_path = os.path.join(output_dir, "indicator_weights.json")
    try:
        with open(weights_file_path, 'w', encoding='utf-8') as f:
            json.dump(weights_dict, f, ensure_ascii=False, indent=4)
        logger.info(f"指标权重已保存到: {weights_file_path}")
    except Exception as e:
        logger.error(f"保存指标权重时出错: {e}")
    
    # 计算综合得分
    all_metrics_df['Score'] = calculate_comprehensive_score(scaler, weights)
    
    # 将归一化后的指标更新到数据框
    for i, name in enumerate(indicator_names):
        all_metrics_df[name] = scaler[:, i]
    
    # 保存完整结果
    all_metrics_df.to_csv(output_file, index=False)
    print(f"所有指标计算完成，结果已保存到 {output_file}")
    
    # 返回完整结果
    return all_metrics_df
