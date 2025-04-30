# 文档指标与综合评分计算工具

## 概述

本项目旨在通过提取文档中的实体信息，计算一系列关键指标（如新颖性、趋势性、适用性等），并采用熵权法动态分配权重，最终生成文档的综合评分。项目支持多进程并行计算及可选的 GPU 加速，显著提升处理效率。

## 主要功能

- **数据预处理**: 加载 CSV 文件，解析日期，处理实体列表。
- **实体库构建**: 提取文档实体，统计词频、出现年份、文档 ID 和共现关系，生成实体知识库（JSON 格式）。支持直接加载现有库。
- **指标计算**: 基于 TF-IDF 和实体库信息，计算以下六大核心指标：
    - 新颖性 (Novelty)
    - 趋势性 (Trend)
    - 适用性 (Applicability)
    - 依赖性 (Dependency)
    - 可替代性 (Replaceability)
    - 成熟度 (Maturity)
- **权重计算**: 使用熵权法动态计算指标权重，并保存为 JSON 文件。
- **综合评分**: 结合归一化后的指标与权重，计算文档的最终综合评分。
- **结果归一化**: 对六大指标及综合评分进行 Min-Max 归一化，映射到 \[0, 1] 区间。
- **增量计算**: 支持加载已有计算结果，仅处理未完成的文档。
- **性能优化**: 提供多进程并行处理及可选的 GPU 加速（主要用于熵权法计算）。
- **灵活配置**: 通过命令行参数配置输入/输出路径、批处理大小、进程数、GPU 使用等。
- **日志记录**: 记录处理过程中的信息、警告和错误，输出到控制台和日志文件。

## 项目结构

```
calculate/
├── data/                            # 数据目录
│   ├── input/                       # 输入文件目录
│   │   ├── extracted_entities.csv   # 示例输入文件 (需用户提供)
│   │   └── stop_words.txt           # 停用词文件 (可选)
│   ├── output/                      # 输出文件目录
│   │   ├── entity_library.json      # 生成的实体库
│   │   └── metrics_results.csv      # 生成的指标结果
│   └── logs/                        # 日志文件目录
│       └── processing.log           # 运行日志
├── src/                             # 源代码目录
│   ├── core/                        # 核心逻辑模块
│   │   ├── data_loader.py           # 数据加载与预处理
│   │   ├── entity_library.py        # 实体库构建与加载
│   │   ├── metrics_calculator.py    # 指标计算协调器
│   │   ├── weight_calculator.py     # 权重与综合得分计算
│   │   └── __init__.py
│   ├── metrics/                     # 各具体指标计算实现
│   │   ├── novelty.py               # 新颖性计算
│   │   ├── trend.py                 # 趋势性计算
│   │   ├── applicability.py         # 适用性计算
│   │   ├── dependency.py            # 依赖性计算
│   │   ├── replaceability.py        # 可替代性计算
│   │   ├── maturity.py              # 成熟度计算
│   │   └── __init__.py
│   ├── rfm                          # RFM 模型计算实现
│   │   ├── rfm_analysis.ipynb       # RFM 模型分析 Jupyter Notebook
│   │   ├── inventor_rfm.csv
│   │   ├── inventor_rfm_segmented.csv
│   ├── config/                      # 配置模块
│   │   └── settings.py              # 全局配置参数
│   │   └── logging_config.py        # 日志配置
│   │   └── __init__.py
│   ├── utils/                       # 工具函数模块
│   │   ├── gpu_utils.py             # GPU 相关工具
│   │   ├── io_utils.py              # 输入输出工具
│   │   ├── parallel_utils.py        # 并行处理工具
│   │   ├── time_utils.py            # 时间处理工具
│   │   ├── visualization.py         # 可视化工具
│   │   ├── data_utils.py            # 数据处理工具
│   │   └── __init__.py
│   └── __init__.py                  # src 包初始化文件
├── tests/                           # 测试代码目录
│   ├── conftest.py                  # Pytest 配置文件和 Fixtures
│   └── test_*.py                    # 各模块的单元测试文件
│   └── __init__.py
├── main.py                          # 主程序入口脚本
├── requirements.txt                 # 项目依赖库列表
├── setup.py                         # 项目安装脚本
├── LICENSE                          # 项目许可证文件
└── README.md                        # 项目说明文档
```

## 安装

1.  **克隆仓库**:
    ```bash
    git clone https://github.com/Ymy0721/calculate.git
    cd calculate
    ```
2.  **创建虚拟环境** (推荐):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
    *注意*: 如果计划使用 GPU 加速，请确保已安装兼容的 CUDA Toolkit 和 cuDNN，并安装支持 GPU 的 `cupy` 版本。

## 使用方法

通过命令行运行 `main.py` 脚本。

**基本用法**:

```bash
python main.py -i data/input/extracted_entities.csv -o data/output --entity_lib data/output/entity_library.json --metrics_output data/output/metrics_results.csv --log_file data/logs/processing.log --num_processes 4 --use_gpu --batch_size 1000
```

常用参数:

`-i`, `--input`: 指定包含提取实体的输入 CSV 文件路径。默认为 `extracted_entities.csv`。CSV 文件应至少包含以下列：
    - **ID**: 文档唯一标识符。
    - **Application Date**: 文档的申请日期，格式为 `YYYYMMDD`。
    - **Extracted Entities**: 分号加空格 (`; `) 分隔的实体字符串。

`-o`, `--output_dir`: 指定输出目录，用于存放实体库、指标结果和日志文件。默认为 `output`。

`--entity_lib`: 指定实体库文件的完整路径。默认为输出目录下的 `entity_library.json`。

`--metrics_output`: 指定指标计算结果 CSV 文件的完整路径。默认为输出目录下的 `metrics_results.csv`。

`--log_dir`: 指定日志文件存放目录。默认为 `logs`。

`-b`, `--batch_size`: 计算指标时的批处理大小。默认为 `32`。

`--num_processes`: 用于构建实体库和计算指标的并行进程数。默认为 `0` (自动选择，通常为 CPU 核心数 - 1)。设置为 `1` 可禁用多进程。

`--use_gpu`: 尝试使用 GPU 加速计算（主要影响熵权法）。需要安装 `cupy`。

`--skip_build`: 如果实体库文件已存在，则跳过构建步骤，直接加载。

`--force_rebuild`: 强制重新构建实体库，即使文件已存在。

`--single_thread`: 强制部分库（如 NumPy/OpenBLAS）在单线程模式下运行，有时可避免多进程冲突。

`--suppress_warnings`: 禁止显示 Python 警告信息。

### 示例

**使用 4 个进程构建实体库并计算指标**:
```bash
python main.py -i data/input/extracted_entities.csv -o data/output --num_processes 4
```

**强制重新构建实体库并使用 GPU**:
```bash
python main.py -i data/input/extracted_entities.csv -o data/output --force_rebuild --use_gpu
```

**跳过实体库构建，直接加载并计算指标**:
```bash
python main.py -i data/input/extracted_entities.csv -o data/output --skip_build
```

## 指标说明

- **新颖性 (Novelty)**: 衡量文档中实体组合相对于历史文档的独特性。
- **趋势性 (Trend)**: 反映文档中实体在近期的活跃程度。
- **适用性 (Applicability)**: 评估文档中实体组合的广泛应用潜力。
- **依赖性 (Dependency)**: 衡量文档中实体之间的关联紧密程度。
- **可替代性 (Replaceability)**: 评估文档中实体组合被其他实体替代的可能性。
- **成熟度 (Maturity)**: 基于实体的出现频率和时间跨度，衡量实体技术的成熟程度。

## 注意事项

1. 输入 CSV 文件需要包含以下列:
    - **ID**: 文档唯一标识符。
    - **Application Date**: 文档的申请日期，格式为 `YYYYMMDD`，多个日期用分号隔开（只取第一个）。
    - **Extracted Entities**: 分号加空格 (`; `) 分隔的实体字符串。

2. 确保有足够的内存，特别是在处理大型数据集和构建实体库时。

3. 多进程和 GPU 加速的效果取决于具体硬件和数据规模。
