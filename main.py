import os
import sys
import argparse
import warnings
import time

# 导入配置
from src.config.settings import (
    INPUT_ENTITIES_PATH, OUTPUT_DIR, ENTITY_LIB_PATH, 
    METRICS_OUTPUT_FILENAME, INDICATORS,
    DEFAULT_BATCH_SIZE, DEFAULT_NUM_PROCESSES, 
    LOGS_DIR  # 添加日志目录导入
)

# 先解析命令行，查找是否有单线程/警告抑制标志
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='计算文档指标和综合得分')
    parser.add_argument('--single_thread', action='store_true', 
                      help='禁用多线程，限制并行计算')
    parser.add_argument('--suppress_warnings', action='store_true',
                      help='禁止显示警告信息')
    
    # 只解析已知的参数，忽略未知参数
    args, _ = parser.parse_known_args()
    
    # 如果指定了单线程模式，设置环境变量
    if args.single_thread:
        print("已启用单线程模式")
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # 如果指定了禁止显示警告
    if args.suppress_warnings:
        print("已禁止显示警告信息")
        warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import traceback

from src.core.data_loader import prepare_data
from src.core.entity_library import build_entity_library, load_entity_library
from src.core.metrics_calculator import calculate_all_metrics
from src.core.weight_calculator import entropy_weight, normalize_indicators, calculate_comprehensive_score
from src.utils.gpu_utils import check_gpu_available
from src.utils.time_utils import Timer, format_time_delta

def setup_logger(log_dir=None):
    """
    配置日志记录器
    
    参数:
    - log_dir: 日志保存目录，None则使用默认目录
    """
    # 使用指定的日志目录，默认使用配置文件中的LOGS_DIR
    log_dir = log_dir or LOGS_DIR
    log_file = os.path.join(log_dir, 'processing.log')
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='计算文档指标和综合得分')
    parser.add_argument('--input', '-i', default=INPUT_ENTITIES_PATH, help='输入CSV文件路径')
    parser.add_argument('--output_dir', '-o', default=OUTPUT_DIR, help='输出目录')
    parser.add_argument('--log_dir', default=LOGS_DIR, help='日志文件保存目录')  # 添加日志目录参数
    parser.add_argument('--entity_lib', default=ENTITY_LIB_PATH, help='实体库保存路径')
    parser.add_argument('--metrics_output', default=METRICS_OUTPUT_FILENAME, help='指标计算结果文件名')
    parser.add_argument('--batch_size', '-b', type=int, default=DEFAULT_BATCH_SIZE, help='批处理大小')
    parser.add_argument('--use_gpu', action='store_true', help='是否使用GPU加速')
    parser.add_argument('--skip_build', action='store_true', 
                        help='跳过构建实体库，直接加载现有实体库文件')
    parser.add_argument('--single_thread', action='store_true', 
                        help='禁用多线程，限制并行计算')
    parser.add_argument('--suppress_warnings', action='store_true',
                        help='禁止显示警告信息')
    parser.add_argument('--num_processes', type=int, default=DEFAULT_NUM_PROCESSES,
                        help='指定使用的进程数，0表示自动选择')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                        help='日志级别')
    
    args = parser.parse_args()
    
    # 创建输出目录（如果不存在）
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志系统 - 使用专门的日志目录
    logger = setup_logger(args.log_dir)
    logger.setLevel(getattr(logging, args.log_level))
    
    # 构建文件路径
    entity_lib_path = args.entity_lib
    metrics_output_path = os.path.join(args.output_dir, args.metrics_output)
    
    # 记录开始时间
    start_time = datetime.now()
    logger.info(f"开始处理: {start_time}")
    
    # 检测GPU可用性
    use_gpu = False
    if args.use_gpu:
        try:
            gpu_available = check_gpu_available(verbose=True)
            if gpu_available:
                use_gpu = True
                logger.info("将使用GPU加速计算")
            else:
                logger.warning("警告: GPU不可用或未正确配置，将使用CPU计算")
        except Exception as e:
            logger.error(f"检测GPU时出错: {e}")
            logger.info("将使用CPU计算")
    
    try:
        # 步骤1: 准备数据
        with Timer("数据加载和预处理") as timer:
            logger.info("步骤1: 加载和预处理数据...")
            df, entities_list = prepare_data(args.input)
            logger.info(f"数据加载完成，共 {len(df)} 条记录，耗时: {format_time_delta(timer.get_elapsed_time())}")
        
        # 步骤2: 构建实体库
        with Timer("实体库处理") as timer:
            if args.skip_build and os.path.exists(entity_lib_path):
                logger.info("跳过构建实体库，直接加载现有文件...")
                entity_lib = load_entity_library(entity_lib_path)
            else:
                logger.info("步骤2: 构建实体库...")
                entity_lib = build_entity_library(df, entity_lib_path, num_processes=args.num_processes)
            
            if entity_lib:
                logger.info(f"实体库构建/加载完成，共 {len(entity_lib)} 个实体，耗时: {format_time_delta(timer.get_elapsed_time())}")
                
                # 检查共现关系是否正确构建
                empty_co_occur = sum(1 for e in entity_lib if not entity_lib[e].get('co_occurrence', {}))
                if empty_co_occur > 0:
                    logger.warning(f"警告: 有 {empty_co_occur} 个实体没有共现关系")
            else:
                logger.error("实体库为空或加载失败")
                return
        
        # 步骤3: 计算指标和综合得分
        with Timer("指标计算") as timer:
            logger.info("步骤3: 计算指标和综合得分...")
            df_metrics = calculate_all_metrics(
                df, entities_list, entity_lib, 
                output_file=metrics_output_path,
                batch_size=args.batch_size,
                num_processes=args.num_processes,
                use_gpu=use_gpu
            )
            logger.info(f"指标计算完成，结果已保存至 {metrics_output_path}，耗时: {format_time_delta(timer.get_elapsed_time())}")
        
        # 记录结束时间并计算总耗时
        end_time = datetime.now()
        processing_time = end_time - start_time
        logger.info(f"处理完成: {end_time}")
        logger.info(f"总耗时: {format_time_delta(processing_time.total_seconds())}")
        logger.info(f"平均每条记录处理时间: {processing_time.total_seconds() / len(df):.4f} 秒")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
