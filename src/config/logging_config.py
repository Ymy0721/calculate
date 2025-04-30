"""
日志配置模块
"""
import os
import logging
from logging.handlers import RotatingFileHandler
import time

from .settings import DEFAULT_LOGS_DIR

def setup_logger(name=None, log_file=None, level=logging.INFO):
    """配置日志记录器"""
    # 确保日志目录存在
    os.makedirs(DEFAULT_LOGS_DIR, exist_ok=True)
    
    # 如果没有指定日志文件，创建一个基于时间的默认文件名
    if log_file is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(DEFAULT_LOGS_DIR, f'process-{timestamp}.log')
    elif not os.path.isabs(log_file):
        log_file = os.path.join(DEFAULT_LOGS_DIR, log_file)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 创建文件处理器
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    return logger
