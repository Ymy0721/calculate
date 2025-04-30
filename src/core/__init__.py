"""
核心功能模块 - 包含项目的核心处理逻辑

此包包括:
- 数据加载与预处理
- 实体库构建与管理 
- 指标计算协调
- 权重计算
"""

# 从各模块导出主要函数
from .data_loader import prepare_data, load_stop_words
from .entity_library import build_entity_library, load_entity_library
from .metrics_calculator import calculate_all_metrics
from .weight_calculator import entropy_weight, normalize_indicators, calculate_comprehensive_score
