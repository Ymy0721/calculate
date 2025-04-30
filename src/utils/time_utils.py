"""
时间工具模块 - 处理时间相关的实用函数
"""
import time
from datetime import datetime, timedelta
from functools import wraps


def get_timestamp(as_string=False):
    """
    获取当前时间戳
    
    参数:
    - as_string: 是否将时间戳格式化为字符串
    
    返回:
    - 时间戳或格式化的时间戳字符串
    """
    now = datetime.now()
    if as_string:
        return now.strftime("%Y%m%d_%H%M%S")
    return now


def format_time_delta(seconds):
    """
    格式化时间差为可读字符串
    
    参数:
    - seconds: 秒数
    
    返回:
    - 格式化后的字符串，如"2小时15分钟30秒"
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}分{seconds:.2f}秒"
    
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{int(hours)}小时{int(minutes)}分{seconds:.2f}秒"
    
    days, hours = divmod(hours, 24)
    return f"{int(days)}天{int(hours)}小时{int(minutes)}分{seconds:.2f}秒"


def timer_decorator(func):
    """
    计时装饰器，测量函数执行时间
    
    用法:
    @timer_decorator
    def your_function():
        pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"函数 {func.__name__} 执行耗时: {format_time_delta(execution_time)}")
        
        return result
    return wrapper


class Timer:
    """
    计时器类，用于测量代码块的执行时间
    
    用法:
    with Timer("操作名称"):
        # 需要计时的代码
    """
    def __init__(self, name="操作"):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        execution_time = self.end_time - self.start_time
        print(f"{self.name}耗时: {format_time_delta(execution_time)}")
        
    def get_elapsed_time(self):
        """获取已经过的时间（秒）"""
        if self.start_time:
            return time.time() - self.start_time
        return 0
