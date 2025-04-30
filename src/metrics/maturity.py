def calculate_maturity(entity_set, entity_lib):
    """
    计算成熟度指标，基于实体频率和时间跨度
    """
    if not entity_set:
        return 0
    
    maturity_sum = 0
    entity_count = 0
    
    for e in entity_set:
        if e in entity_lib:
            entity_count += 1
            info = entity_lib[e]
            freq = info['count']
            years = info['years']
            if years:
                span = max(years) - min(years) + 1
                maturity_sum += freq * span
            
    return maturity_sum / entity_count if entity_count else 0
