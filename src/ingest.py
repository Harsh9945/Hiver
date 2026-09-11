# src/ingest.py
import sys
import os
import json
import time
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

def process_twcs(raw_csv_path='data/raw/twcs.csv', output_dir='data/processed', max_resolved=30000, max_unresolved=5000):
    print(f'Starting ingestion from {raw_csv_path}...')
    start_time = time.time()
    
    cols = ['tweet_id', 'author_id', 'inbound', 'created_at', 'text', 'response_tweet_id', 'in_response_to_tweet_id']
    df = pd.read_csv(raw_csv_path, usecols=cols, low_memory=False)
    print(f'Loaded {len(df):,} raw tweets in {time.time() - start_time:.2f}s.')
    
    df['tweet_id'] = pd.to_numeric(df['tweet_id'], errors='coerce')
    df['in_response_to_tweet_id'] = pd.to_numeric(df['in_response_to_tweet_id'], errors='coerce')
    
    # Fast lookups using python dictionaries
    parent_map = dict(zip(df['tweet_id'], df['in_response_to_tweet_id']))
    author_map = dict(zip(df['tweet_id'], df['author_id']))
    inbound_map = dict(zip(df['tweet_id'], df['inbound']))
    text_map = dict(zip(df['tweet_id'], df['text']))
    time_map = dict(zip(df['tweet_id'], df['created_at']))
    
    # Find all outbound replies by AppleSupport
    apple_outbound = df[df['author_id'] == 'AppleSupport']
    print(f'Total AppleSupport outbound tweets: {len(apple_outbound):,}')
    
    # Trace to root tweet
    def get_root_id(tid):
        curr = tid
        visited = set()
        while curr in parent_map and pd.notna(parent_map[curr]) and curr not in visited:
            visited.add(curr)
            curr = int(parent_map[curr])
        return curr
    
    # Reconstruct resolved cases
    print('Reconstructing resolved cases...')
    resolved_cases = {}
    for _, row in apple_outbound.iterrows():
        agent_tid = int(row['tweet_id'])
        in_resp_to = row['in_response_to_tweet_id']
        if pd.isna(in_resp_to):
            continue
        cust_tid = int(in_resp_to)
        root_tid = get_root_id(cust_tid)
        
        # Verify root tweet was customer inbound
        if root_tid in inbound_map and inbound_map[root_tid]:
            if root_tid not in resolved_cases:
                resolved_cases[root_tid] = {
                    'case_id': str(root_tid),
                    'customer_id': str(author_map.get(root_tid, 'unknown')),
                    'customer_text': str(text_map.get(root_tid, '')),
                    'created_at': str(time_map.get(root_tid, '')),
                    'first_agent_reply': str(row['text']),
                    'agent_replies': [str(row['text'])],
                    'is_resolved': True
                }
            else:
                resolved_cases[root_tid]['agent_replies'].append(str(row['text']))
                
        if len(resolved_cases) >= max_resolved:
            break
            
    print(f'Reconstructed {len(resolved_cases):,} resolved cases.')
    
    # Filter resolved cases with non-empty text
    resolved_list = [
        c for c in resolved_cases.values() 
        if len(c['customer_text'].strip()) > 10 and not c['customer_text'].strip().startswith('http')
    ]
    print(f'Retained {len(resolved_list):,} clean resolved cases.')
    
    # Find unresolved customer tweets mentioning @AppleSupport or @115858
    print('Extracting unresolved customer tweets...')
    unresolved_list = []
    apple_inbound = df[
        df['inbound'] & 
        df['in_response_to_tweet_id'].isna() & 
        df['text'].str.contains('@AppleSupport|@115858', case=False, na=False)
    ]
    for _, row in apple_inbound.iterrows():
        tid = int(row['tweet_id'])
        if tid not in resolved_cases:
            unresolved_list.append({
                'case_id': str(tid),
                'customer_id': str(row['author_id']),
                'customer_text': str(row['text']),
                'created_at': str(row['created_at']),
                'first_agent_reply': '',
                'agent_replies': [],
                'is_resolved': False
            })
            if len(unresolved_list) >= max_unresolved:
                break
    print(f'Extracted {len(unresolved_list):,} unresolved cases.')
    
    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    resolved_df = pd.DataFrame(resolved_list)
    unresolved_df = pd.DataFrame(unresolved_list)
    
    resolved_path = os.path.join(output_dir, 'cases.parquet')
    unresolved_path = os.path.join(output_dir, 'unresolved_cases.parquet')
    
    resolved_df.to_parquet(resolved_path, index=False)
    unresolved_df.to_parquet(unresolved_path, index=False)
    
    # Save manifest
    manifest = {
        'brand': 'AppleSupport',
        'raw_total_tweets': len(df),
        'total_applesupport_outbound': len(apple_outbound),
        'resolved_cases_indexed': len(resolved_df),
        'unresolved_cases_indexed': len(unresolved_df),
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'sample_customer_text': resolved_df.iloc[0]['customer_text'] if len(resolved_df) > 0 else ''
    }
    with open(os.path.join(output_dir, 'corpus_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        
    print(f'Ingestion completed in {time.time() - start_time:.2f}s. Saved to {output_dir}.')
    return resolved_df, unresolved_df

if __name__ == '__main__':
    process_twcs()
