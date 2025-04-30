"""
测试数据加载模块
"""
import pytest
import pandas as pd
import os
import tempfile
from src.core.data_loader import detect_encoding, load_stop_words, prepare_data

def test_detect_encoding():
    """测试编码检测函数"""
    # 创建测试文件
    utf8_content = "这是一个UTF-8编码的测试文件"
    
    with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False) as temp:
        temp.write(utf8_content)
        temp_path = temp.name
    
    try:
        # 测试编码检测
        encoding = detect_encoding(temp_path)
        assert encoding.lower() in ['utf-8', 'utf8'], f"编码应该是UTF-8，而不是{encoding}"
    finally:
        # 清理临时文件
        os.unlink(temp_path)

def test_load_stop_words():
    """测试停用词加载函数"""
    # 创建临时停用词文件
    stop_words_content = "的\n了\n是\n在\n"
    
    with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False) as temp:
        temp.write(stop_words_content)
        temp_path = temp.name
    
    try:
        # 测试停用词加载
        stop_words = load_stop_words(temp_path)
        
        assert isinstance(stop_words, set), "停用词应为集合类型"
        assert len(stop_words) == 4, f"应包含4个停用词，实际包含{len(stop_words)}个"
        assert "的" in stop_words, "应包含停用词'的'"
        assert "了" in stop_words, "应包含停用词'了'"
    finally:
        # 清理临时文件
        os.unlink(temp_path)

def test_prepare_data():
    """测试数据准备函数"""
    # 创建测试CSV文件
    csv_content = """ID,Title,Application Date,Extracted Entities
1,测试标题1,20180102,实体A; 实体B; 实体C
2,测试标题2,20190304,实体B; 实体D; 实体E
3,测试标题3,20200506,实体A; 实体C; 实体F
"""
    
    with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False, suffix='.csv') as temp:
        temp.write(csv_content)
        temp_path = temp.name
    
    try:
        # 测试数据准备
        df, entities_list = prepare_data(temp_path)
        
        # 验证数据帧
        assert isinstance(df, pd.DataFrame), "返回值应为DataFrame"
        assert len(df) == 3, f"应包含3条记录，实际包含{len(df)}条"
        assert 'Year' in df.columns, "应包含Year列"
        assert df['Year'].iloc[0] == 2018, "第一条记录年份应为2018"
        
        # 验证实体列表
        assert len(entities_list) == 3, f"应包含3个实体列表，实际包含{len(entities_list)}个"
        assert entities_list[0] == ['实体A', '实体B', '实体C'], "第一条记录实体不匹配"
    finally:
        # 清理临时文件
        os.unlink(temp_path)
