import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.gpu_utils import get_array_module

def calculate_replaceability(doc_idx, tfidf_matrix, use_gpu=False):
    """
    计算替代性指标，支持GPU加速
    """
    try:
        # 如果矩阵太大，可能需要分批计算相似度
        if hasattr(tfidf_matrix, 'toarray') and tfidf_matrix.shape[0] > 10000:
            # 获取当前文档的向量
            current_doc = tfidf_matrix[doc_idx:doc_idx+1]
            
            # 计算余弦相似度
            sims = cosine_similarity(current_doc, tfidf_matrix).flatten()
            
            # 删除自身相似度
            sims = np.delete(sims, doc_idx)
            
        else:
            # 使用GPU加速（如果可用）
            if use_gpu:
                xp = get_array_module()
                if hasattr(tfidf_matrix, 'toarray'):
                    # 如果是稀疏矩阵，先转为密集矩阵
                    dense_matrix = tfidf_matrix.toarray()
                    gpu_matrix = xp.array(dense_matrix)
                    current_vec = gpu_matrix[doc_idx:doc_idx+1]
                    # 计算余弦相似度
                    norm = xp.linalg.norm(gpu_matrix, axis=1, keepdims=True)
                    norm[norm == 0] = 1  # 避免除零错误
                    normalized = gpu_matrix / norm
                    sims = xp.dot(normalized, normalized[doc_idx].T).flatten()
                    # 删除自身相似度
                    sims = xp.delete(sims, doc_idx)
                    return float(xp.mean(sims))
                
            # 回退到CPU计算
            sims = cosine_similarity(tfidf_matrix[doc_idx:doc_idx+1], tfidf_matrix).flatten()
            sims = np.delete(sims, doc_idx)
        
        return float(np.mean(sims)) if len(sims) > 0 else 0
        
    except Exception as e:
        print(f"替代性计算错误: {e}")
        return 0
