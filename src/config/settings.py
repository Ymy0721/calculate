"""
全局配置文件，集中管理所有可配置参数
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 数据目录路径
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
LOGS_DIR = os.path.join(DATA_DIR, "logs")

# 输入文件路径
INPUT_ENTITIES_PATH = os.path.join(INPUT_DIR, "extracted_entities.csv")
STOP_WORDS_PATH = os.path.join(INPUT_DIR, "stop_words.txt")

# 输出文件名
ENTITY_LIB_FILENAME = "entity_library.json"
ENTITY_LIB_PATH = os.path.join(OUTPUT_DIR, ENTITY_LIB_FILENAME)
METRICS_OUTPUT_FILENAME = "metrics_results.csv"

# 性能配置
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_PROCESSES = 0  # 0表示自动选择

# 处理配置
MIN_ENTITY_FREQUENCY = 2  # 实体最小频率要求

# 指标配置
INDICATORS = ['Novelty', 'Trend', 'Applicability', 'Dependency', 'Replaceability', 'Maturity']

# 其他
RANDOM_SEED = 42

# 确保必要的目录存在
for directory in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)
