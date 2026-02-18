import pandas as pd
import uuid
from collections import defaultdict
from utils.normalization import normalize_text

def union_find_merge(groups):
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x
    
    all_items = set()
    for group in groups:
        all_items.update(group)
        if len(group) > 1:
            items_list = list(group)
            for i in range(1, len(items_list)):
                union(items_list[0], items_list[i])
    
    result_groups = defaultdict(set)
    for item in all_items:
        root = find(item)
        result_groups[root].add(item)
    
    return result_groups

if __name__ == "__main__":
    print("DEDUPLICATING PROFESSORS")
    df = pd.read_csv("processed/br-capes-colsucup-docente.csv", sep=",", low_memory=False, encoding='utf-8')
    print(f"    SIZE: {len(df)} ROWS")
    
    df['index'] = df.index
    
    key1 = (
        df['professor_name'].apply(normalize_text) + "|" + 
        df['professor_document_number'].astype(str).str.strip()
    )
    
    name = df['professor_name'].apply(normalize_text).fillna('')
    birth_year = df['professor_birth_year'].astype(str).str.strip().fillna('')
    doctorate_year = df['professor_degree_year'].astype(str).str.strip().fillna('')
    key2 = name + "|" + birth_year + "|" + doctorate_year
    
    key_to_indices = defaultdict(set)
    key1_indices = defaultdict(set)
    key2_indices = defaultdict(set)
    
    for idx, k1 in enumerate(key1):
        if pd.notna(k1) and k1 and '|' in str(k1) and not str(k1).startswith('nan|'):
            parts = str(k1).split('|')
            if len(parts) == 2 and parts[0] and parts[1] and parts[1] != 'nan':
                key_to_indices[k1].add(idx)
                key1_indices[k1].add(idx)
    
    for idx, k2 in enumerate(key2):
        if pd.notna(k2) and k2 and '|' in str(k2) and not str(k2).startswith('nan|'):
            parts = str(k2).split('|')
            if len(parts) >= 2 and parts[0] and parts[1] and parts[1] != 'nan':
                key_to_indices[k2].add(idx)
                key2_indices[k2].add(idx)
    
    unique_key1 = len([k for k, v in key1_indices.items() if len(v) > 0])
    unique_key2 = len([k for k, v in key2_indices.items() if len(v) > 0])
    print(f"    UNIQUE PROFESSORS (name+document): {unique_key1}")
    print(f"    UNIQUE PROFESSORS (name+birth+degree_year): {unique_key2}")
    
    merged_groups = union_find_merge(key_to_indices.values())
    print(f"    UNIQUE PROFESSORS AFTER MERGE: {len(merged_groups)}")
    
    professor_uuid_map = {}
    for indices in merged_groups.values():
        prof_uuid = uuid.uuid4()
        for idx in indices:
            professor_uuid_map[idx] = prof_uuid
    
    df['professor_id'] = df['index'].map(professor_uuid_map)
    mask = df['professor_id'].isna()
    unmatched_indices = df[mask]['index'].tolist()
    for idx in unmatched_indices:
        professor_uuid_map[idx] = uuid.uuid4()
    df['professor_id'] = df['index'].map(professor_uuid_map)
    
    df = df.drop(columns=['index', 'professor_name', 'professor_document_number', 'professor_birth_year', 'professor_degree_year', 'professor_document_type'])
    unique_professors_final = df['professor_id'].nunique()
    df.to_csv("processed/br-capes-colsucup-docente-professors-deduplicated.csv", sep=",", index=False, encoding='utf-8')
    print(f"\nTOTAL: {len(df)} ROWS")
    print(f"    UNIQUE PROFESSORS FINAL: {unique_professors_final}")
    print("DEDUPLICATION COMPLETED")
