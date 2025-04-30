import numpy as np
from src.utils.gpu_utils import get_array_module

def calculate_novelty(doc_idx, tfidf_matrix, historical_tfidf_mean, use_gpu=False):
    """
    计算新颖性指标，支持GPU加速
    """
    xp = get_array_module() if use_gpu else np
    
    try:
        # 获取当前文档的TF-IDF向量
        if isinstance(tfidf_matrix, np.ndarray) or hasattr(tfidf_matrix, 'toarray'):
            if hasattr(tfidf_matrix, 'toarray'):
                current_vector = tfidf_matrix[doc_idx].toarray().flatten()
            else:
                current_vector = tfidf_matrix[doc_idx].flatten()
            
            # 使用GPU加速（如果可用）
            if use_gpu:
                current_vector = xp.array(current_vector)
                hist_mean = xp.array(historical_tfidf_mean)
            else:
                hist_mean = historical_tfidf_mean
                
            nonzero_indices = xp.where(current_vector > 0)[0]
            if nonzero_indices.size == 0:
                return 0
                
            n_top = max(1, int(0.1 * len(nonzero_indices)))
            top_indices = nonzero_indices[xp.argsort(current_vector[nonzero_indices])[::-1][:n_top]]
            
            # 计算新颖性
            novelty = float(xp.sum(current_vector[top_indices] - hist_mean[top_indices]))
            return novelty
    except Exception as e:
        print(f"新颖性计算错误: {e}")
        return 0
    
    return 0
