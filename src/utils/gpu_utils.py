import warnings
import multiprocessing as mp

# 创建全局标志，避免重复检测
_GPU_DETECTION_DONE = False
_GPU_AVAILABLE = False
_GPU_COUNT = 0

def check_gpu_available(force_check=False, verbose=True):
    """
    检查系统是否有GPU可用，使用全局标志避免重复检测
    
    参数:
    - force_check: 强制重新检测，忽略缓存结果
    - verbose: 是否输出详细信息
    """
    global _GPU_DETECTION_DONE, _GPU_AVAILABLE, _GPU_COUNT
    
    # 如果已经检测过且不需要强制重新检测，直接返回缓存的结果
    if _GPU_DETECTION_DONE and not force_check:
        return _GPU_AVAILABLE
    
    try:
        # 尝试导入cupy
        import cupy as cp
        
        try:
            # 获取设备数量
            num_gpus = cp.cuda.runtime.getDeviceCount()
            
            if num_gpus > 0:
                _GPU_AVAILABLE = True
                _GPU_COUNT = num_gpus
                
                if verbose:
                    print(f"检测到 {num_gpus} 个 GPU 设备")
                    # 列出设备信息
                    for i in range(num_gpus):
                        device = cp.cuda.runtime.getDeviceProperties(i)
                        print(f"  GPU {i}: {device['name']}, 内存: {device['totalGlobalMem'] / (1024**3):.2f} GB")
            else:
                _GPU_AVAILABLE = False
                if verbose:
                    print("未检测到GPU设备")
        except Exception as e:
            _GPU_AVAILABLE = False
            if verbose:
                print(f"GPU初始化错误: {e}")
    except (ImportError, ModuleNotFoundError):
        _GPU_AVAILABLE = False
        if verbose:
            print("未安装cupy，无法使用GPU加速")
            print("可以使用 'pip install cupy-cuda12x' 安装cupy (替换12x为你的CUDA版本)")
    except Exception as e:
        _GPU_AVAILABLE = False
        if verbose:
            print(f"检查GPU时发生错误: {e}")
    
    # 标记检测已完成
    _GPU_DETECTION_DONE = True
    return _GPU_AVAILABLE

def get_array_module():
    """返回适当的数组模块（numpy或cupy）"""
    if check_gpu_available(verbose=False):
        try:
            import cupy as cp
            return cp
        except ImportError:
            pass
    
    import numpy as np
    return np

# 如果在子进程中导入，则不进行GPU检测
if mp.current_process().name == 'MainProcess':
    # 只在主进程中进行GPU检测
    pass
else:
    # 子进程中不输出GPU检测信息
    _GPU_DETECTION_DONE = True  # 假装已经检测过了
