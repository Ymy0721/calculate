"""
IO工具模块 - 处理文件输入输出相关操作
"""
import os
import json
import pickle
import csv
import chardet
import pandas as pd
from pathlib import Path


def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def detect_encoding(file_path, sample_size=10000):
    """
    检测文件编码
    
    参数:
    - file_path: 文件路径
    - sample_size: 采样大小
    
    返回:
    - 编码类型
    """
    with open(file_path, 'rb') as f:
        raw = f.read(sample_size)
        result = chardet.detect(raw)
    
    encoding = result.get('encoding', 'utf-8')
    confidence = result.get('confidence', 0)
    
    # 处理一些特殊情况
    if encoding == 'ascii' or confidence < 0.7:
        return 'utf-8'  # ASCII是UTF-8的子集
    
    return encoding


def safe_read_csv(file_path, **kwargs):
    """
    安全地读取CSV文件，自动处理编码问题
    
    参数:
    - file_path: 文件路径
    - kwargs: 传递给pd.read_csv的其他参数
    
    返回:
    - DataFrame对象
    """
    try:
        # 首先尝试自动检测编码
        encoding = detect_encoding(file_path)
        return pd.read_csv(file_path, encoding=encoding, **kwargs)
    except UnicodeDecodeError:
        # 如果失败，尝试常用编码
        for enc in ['utf-8', 'gbk', 'latin1', 'iso-8859-1']:
            try:
                return pd.read_csv(file_path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        # 所有尝试都失败
        raise ValueError(f"无法确定文件 {file_path} 的编码")


def save_json(data, file_path, ensure_ascii=False, indent=2):
    """
    将数据保存为JSON文件
    
    参数:
    - data: 要保存的数据
    - file_path: 保存路径
    - ensure_ascii: 是否确保ASCII编码
    - indent: 缩进空格数
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)


def load_json(file_path):
    """
    加载JSON文件
    
    参数:
    - file_path: 文件路径
    
    返回:
    - 解析后的JSON数据
    """
    if not os.path.exists(file_path):
        return None
        
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data, file_path):
    """
    将数据保存为Pickle文件
    
    参数:
    - data: 要保存的数据
    - file_path: 保存路径
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path):
    """
    加载Pickle文件
    
    参数:
    - file_path: 文件路径
    
    返回:
    - 反序列化后的数据
    """
    if not os.path.exists(file_path):
        return None
        
    with open(file_path, 'rb') as f:
        return pickle.load(f)
