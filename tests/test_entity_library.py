"""
测试实体库模块
"""
import pytest
import os
import tempfile
import json
from src.core.entity_library import build_entity_library, load_entity_library

def test_load_entity_library():
    """测试加载实体库函数"""
    # 创建临时JSON文件模拟实体库
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as temp:
        # 写入示例数据
        sample_lib = {
            "entity1": {
                "count": 3,
                "years": [2018, 2020, 2022],
                "doc_ids": ["1", "3", "5"],
                "co_occurrence": {"entity2": 1, "entity3": 1}
            },
            "entity2": {
                "count": 2,
                "years": [2018, 2019],
                "doc_ids": ["1", "2"],
                "co_occurrence": {"entity1": 1, "entity4": 1}
            }
        }
        json.dump(sample_lib, temp)
        temp_path = temp.name
    
    try:
        # 测试加载
        loaded_lib = load_entity_library(temp_path)
        
        # 验证加载结果
        assert loaded_lib is not None, "应成功加载实体库"
        assert isinstance(loaded_lib, dict), "实体库应该是字典"
        assert "entity1" in loaded_lib, "应包含entity1"
        assert "entity2" in loaded_lib, "应包含entity2"
        assert loaded_lib["entity1"]["count"] == 3, "entity1的计数应为3"
        assert loaded_lib["entity2"]["count"] == 2, "entity2的计数应为2"
        
        # 测试非法路径
        with pytest.raises(FileNotFoundError):
            load_entity_library("non_existent_file.json")
            
    finally:
        # 清理临时文件
        os.unlink(temp_path)

def test_build_entity_library(sample_df):
    """测试构建实体库函数"""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as temp:
        temp_path = temp.name
    
    try:
        # 构建实体库
        entity_lib = build_entity_library(sample_df, temp_path, force_rebuild=True, num_processes=1)
        
        # 验证结果
        assert entity_lib is not None, "应成功构建实体库"
        assert isinstance(entity_lib, dict), "实体库应该是字典"
        
        # 检查是否包含所有实体
        expected_entities = {
            'entity1', 'entity2', 'entity3', 'entity4', 'entity5', 
            'entity6', 'entity7', 'entity8', 'entity9'
        }
        for entity in expected_entities:
            assert entity in entity_lib, f"{entity}应包含在实体库中"
            
        # 验证计数
        assert entity_lib['entity1']['count'] >= 1, "entity1的计数至少为1"
        
        # 检查共现矩阵存在
        assert 'co_occurrence' in entity_lib['entity1'], "应包含共现矩阵"
        
        # 验证文件创建
        assert os.path.exists(temp_path), "应创建实体库文件"
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)
