"""
pytest配置文件 - 定义测试夹具和共享资源
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# 确保src目录在Python路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture
def sample_entities_list():
    """提供示例实体列表用于测试"""
    return [
        ['实体1', '实体2', '实体3'],
        ['实体2', '实体4', '实体5'],
        ['实体1', '实体5', '实体6'],
        ['实体3', '实体6', '实体7'],
        ['实体1', '实体8', '实体9'],
    ]

@pytest.fixture
def sample_df():
    """提供示例数据帧用于测试"""
    data = {
        'ID': [1, 2, 3, 4, 5],
        'Title': [f'标题{i}' for i in range(1, 6)],
        'Year': [2018, 2019, 2020, 2021, 2022],
        'Extracted Entities': [
            'entity1; entity2; entity3',
            'entity2; entity4; entity5',
            'entity1; entity5; entity6',
            'entity3; entity6; entity7',
            'entity1; entity8; entity9',
        ]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_entity_lib():
    """提供示例实体库用于测试"""
    entity_lib = {
        'entity1': {
            'count': 3,
            'years': [2018, 2020, 2022],
            'doc_ids': ['1', '3', '5'],
            'co_occurrence': {'entity2': 1, 'entity3': 1, 'entity5': 1, 'entity6': 1, 'entity8': 1, 'entity9': 1}
        },
        'entity2': {
            'count': 2,
            'years': [2018, 2019],
            'doc_ids': ['1', '2'],
            'co_occurrence': {'entity1': 1, 'entity3': 1, 'entity4': 1, 'entity5': 1}
        },
        'entity3': {
            'count': 2,
            'years': [2018, 2021],
            'doc_ids': ['1', '4'],
            'co_occurrence': {'entity1': 1, 'entity2': 1, 'entity6': 1, 'entity7': 1}
        },
        'entity4': {
            'count': 1,
            'years': [2019],
            'doc_ids': ['2'],
            'co_occurrence': {'entity2': 1, 'entity5': 1}
        },
        'entity5': {
            'count': 2,
            'years': [2019, 2020],
            'doc_ids': ['2', '3'],
            'co_occurrence': {'entity1': 1, 'entity2': 1, 'entity4': 1, 'entity6': 1}
        },
        'entity6': {
            'count': 2,
            'years': [2020, 2021],
            'doc_ids': ['3', '4'],
            'co_occurrence': {'entity1': 1, 'entity3': 1, 'entity5': 1, 'entity7': 1}
        },
        'entity7': {
            'count': 1,
            'years': [2021],
            'doc_ids': ['4'],
            'co_occurrence': {'entity3': 1, 'entity6': 1}
        },
        'entity8': {
            'count': 1,
            'years': [2022],
            'doc_ids': ['5'],
            'co_occurrence': {'entity1': 1, 'entity9': 1}
        },
        'entity9': {
            'count': 1,
            'years': [2022],
            'doc_ids': ['5'],
            'co_occurrence': {'entity1': 1, 'entity8': 1}
        }
    }
    return entity_lib

@pytest.fixture
def sample_tfidf_matrix():
    """提供示例TF-IDF矩阵用于测试"""
    # 创建一个示例TF-IDF矩阵，5个文档，9个特征
    data = np.array([
        [0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.4, 0.0, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0],
        [0.4, 0.0, 0.0, 0.0, 0.3, 0.3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.4, 0.0, 0.0, 0.3, 0.3, 0.0, 0.0],
        [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3]
    ])
    return data
