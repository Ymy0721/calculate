"""
测试指标计算模块
"""
import pytest
import numpy as np
from src.metrics.novelty import calculate_novelty
from src.metrics.trend import calculate_trend
from src.metrics.applicability import calculate_applicability
from src.metrics.dependency import calculate_dependency
from src.metrics.replaceability import calculate_replaceability
from src.metrics.maturity import calculate_maturity


def test_novelty(sample_tfidf_matrix):
    """测试新颖性指标计算"""
    # 计算历史TF-IDF均值
    historical_mean = np.mean(sample_tfidf_matrix, axis=0)
    
    # 测试第一个文档的新颖性
    novelty = calculate_novelty(0, sample_tfidf_matrix, historical_mean, False)
    
    # 验证结果
    assert isinstance(novelty, (int, float)), "新颖性应该是数值"
    assert not np.isnan(novelty), "新颖性不应该是NaN"


def test_trend(sample_entity_lib):
    """测试趋势指标计算"""
    # 测试包含多个实体的集合
    entity_set = {'entity1', 'entity2', 'entity3'}
    trend = calculate_trend(entity_set, sample_entity_lib)
    
    # 验证结果
    assert isinstance(trend, (int, float)), "趋势应该是数值"
    assert not np.isnan(trend), "趋势不应该是NaN"
    
    # 测试空集
    empty_trend = calculate_trend(set(), sample_entity_lib)
    assert empty_trend == 0, "空实体集的趋势应该是0"


def test_applicability(sample_entity_lib, sample_tfidf_matrix):
    """测试适用性指标计算"""
    # 测试包含多个实体的集合
    entity_set = {'entity1', 'entity2', 'entity5'}
    applicability = calculate_applicability(entity_set, sample_entity_lib, sample_tfidf_matrix, False)
    
    # 验证结果
    assert isinstance(applicability, (int, float)), "适用性应该是数值"
    assert 0 <= applicability <= 1, "适用性应该在0到1之间"
    
    # 测试空集
    empty_applicability = calculate_applicability(set(), sample_entity_lib, sample_tfidf_matrix, False)
    assert empty_applicability == 0, "空实体集的适用性应该是0"


def test_dependency(sample_entity_lib):
    """测试依赖性指标计算"""
    # 测试包含多个实体的集合
    entity_set = {'entity1', 'entity2', 'entity3'}
    dependency = calculate_dependency(entity_set, sample_entity_lib)
    
    # 验证结果
    assert isinstance(dependency, (int, float)), "依赖性应该是数值"
    assert dependency >= 0, "依赖性应该是非负数"
    
    # 测试空集
    empty_dependency = calculate_dependency(set(), sample_entity_lib)
    assert empty_dependency == 0, "空实体集的依赖性应该是0"


def test_replaceability(sample_tfidf_matrix):
    """测试替代性指标计算"""
    # 测试文档替代性
    replaceability = calculate_replaceability(0, sample_tfidf_matrix, False)
    
    # 验证结果
    assert isinstance(replaceability, (int, float)), "替代性应该是数值"
    assert 0 <= replaceability <= 1, "替代性应该在0到1之间"


def test_maturity(sample_entity_lib):
    """测试成熟度指标计算"""
    # 测试包含多个实体的集合
    entity_set = {'entity1', 'entity5', 'entity6'}
    maturity = calculate_maturity(entity_set, sample_entity_lib)
    
    # 验证结果
    assert isinstance(maturity, (int, float)), "成熟度应该是数值"
    assert maturity >= 0, "成熟度应该是非负数"
    
    # 测试空集
    empty_maturity = calculate_maturity(set(), sample_entity_lib)
    assert empty_maturity == 0, "空实体集的成熟度应该是0"
