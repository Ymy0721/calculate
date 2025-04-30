def calculate_dependency(entity_set, entity_lib):
    """
    计算依赖性指标，基于共现关系
    
    参数:
    - entity_set: 当前文档的实体集合
    - entity_lib: 实体库
    
    返回:
    - 依赖性指标值
    """
    if not entity_set:
        return 0
    
    dependency = 0
    for e in entity_set:
        if e in entity_lib:
            # 直接计算共现关系的总和
            co_occur = sum(entity_lib[e].get('co_occurrence', {}).values())
            dependency += co_occur
    
    # 计算平均依赖性
    return dependency / len(entity_set) if entity_set else 0