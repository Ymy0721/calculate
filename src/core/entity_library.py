import json
import os
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from .data_loader import load_stop_words

# 多进程处理配置
NUM_PROCESSES = max(1, mp.cpu_count() - 1)  # 预留一个核心给系统

def load_entity_library(filepath):
    """直接加载已有的实体库"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"实体库文件 '{filepath}' 不存在")
        
    print(f"正在加载现有实体库: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            entity_lib = json.load(f)
        print(f"成功加载实体库，包含 {len(entity_lib)} 个实体")
        return entity_lib
    except Exception as e:
        print(f"加载实体库时出错: {e}")
        return None

def process_entities_chunk(chunk):
    """处理实体块并返回局部结果"""
    index_range, entities_list, years, doc_ids = chunk
    local_count = {}
    local_years = {}
    local_docids = {}
    local_co_occur = {}

    for i, entities in enumerate(entities_list):
        if not isinstance(entities, list) or not entities:
            continue
            
        current_docid = doc_ids[i]
        unique_entities = set(entities)
        
        # 过滤掉空实体
        unique_entities = [e for e in unique_entities if e and not e.isspace()]
        
        if not unique_entities:
            continue
            
        # 统计实体基础信息
        for e in unique_entities:
            if e not in local_count:
                local_count[e] = 0
                local_years[e] = set()
                local_docids[e] = set()
                
            local_count[e] += 1
            local_years[e].add(years[i])
            local_docids[e].add(current_docid)

            # 初始化共现计数
            if e not in local_co_occur:
                local_co_occur[e] = {}

        # 统计共现
        entities_in_doc = list(unique_entities)
        for j in range(len(entities_in_doc)):
            e1 = entities_in_doc[j]
            for k in range(j + 1, len(entities_in_doc)):
                e2 = entities_in_doc[k]
                if e1 != e2:  # 确保不与自己共现
                    local_co_occur[e1][e2] = local_co_occur[e1].get(e2, 0) + 1
                    local_co_occur[e2][e1] = local_co_occur[e2].get(e1, 0) + 1

    return local_count, local_years, local_docids, local_co_occur, len(index_range)

def build_entity_library(df, output_lib='entity_library.json', force_rebuild=False, num_processes=0):
    """构建实体库（包含共现）或直接加载已有实体库"""
    # 如果实体库文件已存在且不强制重建，则直接加载
    if os.path.exists(output_lib) and not force_rebuild:
        print("检测到已存在的实体库，直接加载...")
        return load_entity_library(output_lib)
    
    print("正在构建新的实体库...")
    
    # 设置进程数
    if num_processes <= 0:
        process_count = max(1, mp.cpu_count() - 1)
    else:
        process_count = num_processes
        print(f"指定使用 {process_count} 个进程")
        
    stop_words = load_stop_words()
    entities_list = df['Extracted Entities'].fillna('').str.split('; ').tolist()
    years = df['Year'].tolist()
    doc_ids = df['ID'].tolist()

    num_records = len(df)
    chunk_size = (num_records + process_count - 1) // process_count
    chunks = [
        (
            range(i * chunk_size, min((i + 1) * chunk_size, num_records)),
            entities_list[i * chunk_size: min((i + 1) * chunk_size, num_records)],
            years[i * chunk_size: min((i + 1) * chunk_size, num_records)],
            doc_ids[i * chunk_size: min((i + 1) * chunk_size, num_records)]
        )
        for i in range(process_count)
    ]

    # 使用多进程处理
    with tqdm(total=num_records, desc="构建实体库...") as pbar:
        with mp.Pool(process_count) as pool:
            results = []
            for result in pool.imap_unordered(process_entities_chunk, chunks):
                results.append(result)
                pbar.update(result[-1])  # 更新进度条

    # 合并所有结果
    entity_count = {}
    entity_years = {}
    entity_docids = {}
    co_occurrence = {}

    for local_count, local_years, local_docids, local_co_occur, _ in results:
        # 合并基础信息
        for e, count in local_count.items():
            if e not in entity_count:
                entity_count[e] = 0
                entity_years[e] = set()
                entity_docids[e] = set()
            entity_count[e] += count
            entity_years[e].update(local_years.get(e, set()))
            entity_docids[e].update(local_docids.get(e, set()))

        # 合并共现信息
        for e1, co_dict in local_co_occur.items():
            if e1 not in co_occurrence:
                co_occurrence[e1] = {}
            for e2, cnt in co_dict.items():
                co_occurrence[e1][e2] = co_occurrence[e1].get(e2, 0) + cnt

    # 过滤共现关系，只保留出现至少两次的共现
    for e1 in list(co_occurrence.keys()):
        co_occurrence[e1] = {e2: cnt for e2, cnt in co_occurrence[e1].items() if cnt >= 2}
        if not co_occurrence[e1]:  # 删除空字典
            del co_occurrence[e1]

    # 添加调试信息
    empty_co_occur_before = sum(1 for e in entity_count if e not in co_occurrence)
    total_entities = len(entity_count)

    # 转换为常规字典并过滤
    final_entity_lib = {}
    for e in entity_count:
        if e in stop_words or not e or e.isspace():
            continue
            
        final_entity_lib[e] = {
            "count": int(entity_count[e]),
            "years": sorted(entity_years[e]),
            "doc_ids": sorted(map(str, entity_docids[e])),  # 转换为字符串便于序列化
            "co_occurrence": dict(co_occurrence.get(e, {}))
        }

    # 输出调试信息
    print(f"共有 {total_entities} 个实体，其中 {empty_co_occur_before} 个实体没有共现关系 (原始统计)")

    # 过滤低频实体
    filtered_entity_lib = {k: v for k, v in final_entity_lib.items() if v['count'] >= 2}
    sorted_entity_lib = dict(sorted(filtered_entity_lib.items(),
                                    key=lambda item: item[1]['count'],
                                    reverse=True))

    # 统计过滤后的结果
    empty_co_occur_after = sum(1 for e in sorted_entity_lib if not sorted_entity_lib[e]['co_occurrence'])
    print(f"过滤后共有 {len(sorted_entity_lib)} 个实体，其中 {empty_co_occur_after} 个实体没有共现关系")

    # 保存为JSON
    with open(output_lib, 'w', encoding='utf-8') as f:
        json.dump(sorted_entity_lib, f, ensure_ascii=False, indent=2)

    return sorted_entity_lib
