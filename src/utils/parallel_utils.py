"""
并行处理工具模块 - 提供多进程和多线程执行的工具函数
"""
import os
import sys
import multiprocessing as mp
import concurrent.futures
from functools import partial
from tqdm import tqdm


def get_optimal_worker_count(requested_workers=0, reserve_cores=1):
    """
    获取最佳工作进程/线程数
    
    参数:
    - requested_workers: 请求的工作线程/进程数，0表示自动选择
    - reserve_cores: 预留的核心数
    
    返回:
    - 建议使用的工作线程/进程数
    """
    if requested_workers > 0:
        return requested_workers
    
    # 获取CPU核心数
    cpu_count = mp.cpu_count()
    
    # 保留至少一个核心给系统和主线程
    return max(1, cpu_count - reserve_cores)


def parallel_process(func, items, n_workers=None, use_threads=False, 
                     show_progress=True, desc="处理中", **kwargs):
    """
    并行处理一组项目
    
    参数:
    - func: 要处理每个项目的函数
    - items: 项目列表
    - n_workers: 工作进程/线程数，None表示自动选择
    - use_threads: 是否使用线程而非进程
    - show_progress: 是否显示进度条
    - desc: 进度条描述
    - kwargs: 传递给进度条的其他参数
    
    返回:
    - 结果列表
    """
    if n_workers is None:
        n_workers = get_optimal_worker_count()
    
    executor_class = concurrent.futures.ThreadPoolExecutor if use_threads else concurrent.futures.ProcessPoolExecutor
    
    results = []
    with executor_class(max_workers=n_workers) as executor:
        futures = [executor.submit(func, item) for item in items]
        
        if show_progress:
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=desc, **kwargs):
                results.append(future.result())
        else:
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    
    return results


def chunk_list(lst, n):
    """
    将列表分成n个大致相等的块
    
    参数:
    - lst: 要分块的列表
    - n: 分块数
    
    返回:
    - 分块后的列表的列表
    """
    chunk_size = max(1, len(lst) // n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def map_reduce_parallel(map_func, reduce_func, items, n_workers=None, 
                        show_progress=True, desc="Map-Reduce处理中"):
    """
    并行执行Map-Reduce操作
    
    参数:
    - map_func: 映射函数
    - reduce_func: 归约函数
    - items: 项目列表
    - n_workers: 工作进程数，None表示自动选择
    - show_progress: 是否显示进度条
    - desc: 进度条描述
    
    返回:
    - 归约后的结果
    """
    if n_workers is None:
        n_workers = get_optimal_worker_count()
    
    # 防止进程数超过项目数
    n_workers = min(n_workers, len(items))
    
    if n_workers <= 1:
        # 单进程直接处理
        items_to_iterate = tqdm(items, desc=desc) if show_progress else items
        mapped = [map_func(item) for item in items_to_iterate]
        return reduce_func(mapped)
    
    # 分块
    chunks = chunk_list(items, n_workers)
    
    # 部分映射函数：对一个块应用map_func
    def partial_map(chunk):
        return [map_func(item) for item in chunk]
    
    # 并行执行部分映射
    with mp.Pool(n_workers) as pool:
        if show_progress:
            mapped_chunks = list(tqdm(pool.imap(partial_map, chunks), total=len(chunks), desc=desc))
        else:
            mapped_chunks = pool.map(partial_map, chunks)
    
    # 合并映射结果
    all_mapped = [item for chunk in mapped_chunks for item in chunk]
    
    # 归约结果
    return reduce_func(all_mapped)
