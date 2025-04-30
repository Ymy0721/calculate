import pandas as pd
import numpy as np
import chardet
from datetime import datetime
import warnings
import os

from src.config.settings import STOP_WORDS_PATH

def detect_encoding(file_path, sample_size=100):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(sample_size))
    return result['encoding']

def load_data(file_path):
    """加载CSV数据并处理编码问题"""
    # 检测编码
    encoding = detect_encoding(file_path)
    print(f"检测到编码: {encoding}")
    
    # 尝试使用检测到的编码加载
    try:
        df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
        print(f"使用 {encoding} 编码成功加载数据。")
    except UnicodeDecodeError:
        # 如果失败，尝试常用编码
        for enc in ['utf-8', 'gbk', 'latin1']:
            try:
                df = pd.read_csv(file_path, encoding=enc, on_bad_lines='skip')
                print(f"使用 {enc} 编码成功加载数据。")
                break
            except UnicodeDecodeError:
                continue
    
    # 统计被忽略的行数
    total_rows = sum(1 for line in open(file_path, 'rb'))
    loaded_rows = len(df)
    skipped_rows = total_rows - loaded_rows - 1  # 减去表头
    print(f"文件总行数: {total_rows}")
    print(f"已加载行数: {loaded_rows}")
    print(f"因解码错误跳过的行数: {skipped_rows}")
    
    return df

def preprocess_dates(df):
    """预处理申请日期"""
    df['First_Date'] = df['Application Date'].astype(str).str.split(';').str[0]
    df['Clean_Date'] = df['First_Date'].str.replace(r'[^0-9.]', '', regex=True)
    df['Application Date'] = pd.to_datetime(df['Clean_Date'], format='%Y%m%d', errors='coerce')
    
    invalid_count = df['Application Date'].isna().sum()
    if invalid_count > 0:
        print(f"{invalid_count} 条无效日期记录已删除")
    
    df = df.dropna(subset=['Application Date']).reset_index(drop=True)
    df['Year'] = df['Application Date'].dt.year
    return df.drop(columns=['First_Date', 'Clean_Date'])

def load_stop_words(filepath=None):
    """加载停用词"""
    if filepath is None:
        filepath = STOP_WORDS_PATH
        
    stop_words = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            stop_words.update(line.strip() for line in f if line.strip())
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="gbk", errors='ignore') as f:
            stop_words.update(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        print(f"警告: 未找到停用词文件 {filepath}")
    return stop_words

def prepare_data(input_file):
    """准备数据集，包括加载、预处理和获取实体列表"""
    # 加载数据
    df = load_data(input_file)
    
    # 日期预处理
    df = preprocess_dates(df)
    
    # 确保ID列存在
    if 'ID' not in df.columns:
        df['ID'] = df.index
        
    # 获取实体列表
    entities_list = df['Extracted Entities'].fillna('').str.split('; ').tolist()
    
    return df, entities_list
