import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.gpu_utils import get_array_module

def calculate_applicability(entity_set, entity_lib, tfidf_matrix, use_gpu=False):
    """
    计算适用性指标（优化版本），支持GPU加速
    
    参数:
    - entity_set: 当前文档的实体集合
    - entity_lib: 实体库
    - tfidf_matrix: TF-IDF矩阵
    - use_gpu: 是否使用GPU加速
    """
    if not entity_set:
        return 0
        
    # 获取适当的数组模块
    xp = get_array_module() if use_gpu else np
    
    applicability_scores = []
    for e in entity_set:
        if e in entity_lib and 'doc_ids' in entity_lib[e]:
            try:
                # 获取文档子集的索引
                doc_ids = [int(i) for i in entity_lib[e]['doc_ids'] if int(i) < tfidf_matrix.shape[0]]
                if len(doc_ids) >= 2:
                    # 获取文档子集的TF-IDF表示
                    tfidf_subset = tfidf_matrix[doc_ids]
                    
                    # 计算文档之间的余弦相似度
                    if use_gpu and hasattr(xp, 'linalg'):
                        # GPU加速计算相似度
                        if hasattr(tfidf_subset, 'toarray'):
                            subset_dense = tfidf_subset.toarray()
                            subset_gpu = xp.array(subset_dense)
                        else:
                            subset_gpu = xp.array(tfidf_subset)
                            
                        # 归一化向量
                        norms = xp.linalg.norm(subset_gpu, axis=1, keepdims=True)
                        norms[norms == 0] = 1  # 避免除零错误
                        normed_vecs = subset_gpu / norms
                        
                        # 计算相似度矩阵
                        similarity_matrix = xp.dot(normed_vecs, normed_vecs.T)
                        
                        # 提取上三角部分
                        mask = xp.triu_indices_from(similarity_matrix, k=1)
                        upper_values = similarity_matrix[mask]
                        
                        avg_similarity = float(xp.mean(upper_values))
                    else:
                        # 使用sklearn的余弦相似度
                        if isinstance(tfidf_subset, csr_matrix):
                            similarity_matrix = cosine_similarity(tfidf_subset)
                        else:
                            similarity_matrix = cosine_similarity(tfidf_subset)
                        
                        # 计算平均相似度
                        upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
                        avg_similarity = np.mean(similarity_matrix[upper_triangle_indices])
                    
                    # 使用相似度的倒数作为分散度的近似
                    applicability = 1 - avg_similarity
                    applicability_scores.append(applicability)
            except Exception as e:
                print(f"计算适用性指标时出错: {e}")
                continue

    return np.mean(applicability_scores) if applicability_scores else 0
