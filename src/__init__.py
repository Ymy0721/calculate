"""
项目主包 - AI-Project的计算分析模块

此包包含用于计算文档指标和综合得分的所有组件。
"""

__version__ = '0.1.0'
__author__ = 'Mingyu Yuan'

# 导入常用模块，便于用户使用
from .core.data_loader import prepare_data
from .core.entity_library import build_entity_library, load_entity_library
from .core.metrics_calculator import calculate_all_metrics
from .utils.gpu_utils import check_gpu_available
