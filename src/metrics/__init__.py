"""
指标计算模块 - 包含各种指标的计算方法

支持的指标:
- 新颖性 (Novelty)
- 趋势性 (Trend)
- 适用性 (Applicability)
- 依赖性 (Dependency)
- 可替代性 (Replaceability)
- 成熟度 (Maturity)
"""

# 导出所有指标计算函数
from .novelty import calculate_novelty
from .trend import calculate_trend
from .applicability import calculate_applicability
from .dependency import calculate_dependency
from .replaceability import calculate_replaceability
from .maturity import calculate_maturity

# 全部指标列表
METRICS = [
    'Novelty',
    'Trend',
    'Applicability', 
    'Dependency',
    'Replaceability',
    'Maturity'
]
