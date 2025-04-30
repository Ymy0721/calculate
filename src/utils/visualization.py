"""
可视化工具模块 - 提供数据可视化和结果展示功能
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns

# 设置中文字体支持
def setup_chinese_font():
    """设置支持中文字体的显示"""
    try:
        font_paths = [
            # Windows 字体路径
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/msyh.ttf',
            # Linux 字体路径
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            # macOS 字体路径
            '/System/Library/Fonts/PingFang.ttc',
        ]
        
        # 查找第一个存在的字体文件
        for font_path in font_paths:
            if os.path.exists(font_path):
                chinese_font = FontProperties(fname=font_path)
                plt.rcParams['font.family'] = chinese_font.get_name()
                return chinese_font
                
        # 如果找不到，使用系统默认配置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        return None
    except Exception as e:
        print(f"设置中文字体时出错: {e}")
        return None

# 初始化中文字体
chinese_font = setup_chinese_font()

def plot_indicators_radar(indicators_data, title="指标雷达图", save_path=None):
    """
    生成指标雷达图
    
    参数:
    - indicators_data: 字典，键为指标名称，值为指标值
    - title: 图表标题
    - save_path: 保存路径，None表示不保存
    
    返回:
    - matplotlib figure对象
    """
    categories = list(indicators_data.keys())
    values = list(indicators_data.values())
    
    # 处理数据使其成为闭环
    categories = categories + [categories[0]]
    values = values + [values[0]]
    
    # 计算角度
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # 绘制线条和填充
    ax.plot(angles, values, 'o-', linewidth=2, label='指标值')
    ax.fill(angles, values, alpha=0.25)
    
    # 设置角度刻度和标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], fontproperties=chinese_font)
    
    # 添加标题
    ax.set_title(title, fontproperties=chinese_font, fontsize=15)
    
    # Y轴刻度
    ax.set_ylim(0, max(values) * 1.1)
    
    # 调整网格线
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_correlation_heatmap(df, columns=None, title="特征相关性热力图", save_path=None):
    """
    生成相关性热力图
    
    参数:
    - df: DataFrame对象
    - columns: 要分析的列，None表示所有数值列
    - title: 图表标题
    - save_path: 保存路径，None表示不保存
    
    返回:
    - matplotlib figure对象
    """
    # 准备数据
    if columns is None:
        # 选择所有数值型列
        data = df.select_dtypes(include=[np.number])
    else:
        data = df[columns]
    
    # 计算相关系数矩阵
    corr = data.corr()
    
    # 设置图表大小
    plt.figure(figsize=(12, 10))
    
    # 生成热力图
    mask = np.triu(np.ones_like(corr, dtype=bool))  # 生成上三角掩码
    heatmap = sns.heatmap(
        corr, 
        annot=True,       # 显示数据值
        fmt=".2f",        # 数据格式
        cmap='coolwarm',  # 色彩映射
        mask=mask,        # 应用掩码
        vmin=-1, vmax=1,  # 值范围
        square=True,      # 正方形单元格
        linewidths=.5,    # 分隔线宽度
        cbar_kws={"shrink": .8}  # 颜色条参数
    )
    
    # 设置标题和轴标签
    plt.title(title, fontproperties=chinese_font, fontsize=16)
    plt.xticks(rotation=45, ha='right', fontproperties=chinese_font)
    plt.yticks(fontproperties=chinese_font)
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_metrics_distribution(df, metric_column, bins=20, title=None, save_path=None):
    """
    生成指标分布直方图
    
    参数:
    - df: DataFrame对象
    - metric_column: 指标列名
    - bins: 柱状图箱数
    - title: 图表标题，None表示自动生成
    - save_path: 保存路径，None表示不保存
    
    返回:
    - matplotlib figure对象
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图和核密度估计
    sns.histplot(df[metric_column], kde=True, bins=bins)
    
    # 设置标题
    if title is None:
        title = f"{metric_column} 分布图"
    plt.title(title, fontproperties=chinese_font, fontsize=14)
    
    # 设置标签
    plt.xlabel(metric_column, fontproperties=chinese_font)
    plt.ylabel("频率", fontproperties=chinese_font)
    
    # 添加统计信息
    mean_val = df[metric_column].mean()
    median_val = df[metric_column].median()
    std_val = df[metric_column].std()
    
    plt.axvline(mean_val, color='r', linestyle='--', label=f'均值: {mean_val:.2f}')
    plt.axvline(median_val, color='g', linestyle='-.', label=f'中位数: {median_val:.2f}')
    
    plt.legend(prop=chinese_font)
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_top_n_entities(df, entity_column, value_column, n=20, title=None, save_path=None):
    """
    绘制Top N实体图表
    
    参数:
    - df: DataFrame对象
    - entity_column: 实体列名
    - value_column: 值列名
    - n: 显示的实体数量
    - title: 图表标题，None表示自动生成
    - save_path: 保存路径，None表示不保存
    
    返回:
    - matplotlib figure对象
    """
    # 选取前N个实体
    top_n = df.sort_values(by=value_column, ascending=False).head(n)
    
    plt.figure(figsize=(12, 8))
    
    # 创建水平条形图
    bars = plt.barh(top_n[entity_column], top_n[value_column])
    
    # 添加数据标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2f}', ha='left', va='center')
    
    # 设置标题
    if title is None:
        title = f"Top {n} {entity_column} by {value_column}"
    plt.title(title, fontproperties=chinese_font, fontsize=16)
    
    # 设置标签
    plt.xlabel(value_column, fontproperties=chinese_font)
    plt.ylabel(entity_column, fontproperties=chinese_font)
    
    # Y轴逆序，让最高的值在顶部
    plt.gca().invert_yaxis()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()
