# tests/test_leakage.py
import json
import pickle
import os

def test_data_leakage_and_integrity():
    with open("data/dev/dev_set.json", "r", encoding="utf-8") as f:
        dev_cases = json.load(f)
    with open("data/golden/golden_set.json", "r", encoding="utf-8") as f:
        golden_cases = json.load(f)
    with open("data/golden/judge_agreement_subset.json", "r", encoding="utf-8") as f:
        agreement_cases = json.load(f)
    with open("data/processed/retrieval_index.pkl", "rb") as f:
        index_bundle = pickle.load(f)
        
    dev_ids = {str(c["case_id"]) for c in dev_cases}
    golden_ids = {str(c["case_id"]) for c in golden_cases}
    agreement_ids = {str(c["case_id"]) for c in agreement_cases}
    index_ids = {str(cid) for cid in index_bundle["case_ids"]}
    
    assert len(golden_ids) == 200, f"Expected 200 golden cases, got {len(golden_ids)}"
    assert len(dev_ids) == 35, f"Expected 35 dev cases, got {len(dev_ids)}"
    assert len(agreement_ids) == 40, f"Expected 40 agreement cases, got {len(agreement_ids)}"
    
    # 1. Dev vs Golden disjoint
    dev_gold_overlap = dev_ids.intersection(golden_ids)
    assert len(dev_gold_overlap) == 0, f"Dev and Golden sets overlap on IDs: {dev_gold_overlap}"
    
    # 2. Golden vs Retrieval Index disjoint (STRICT LEAKAGE CHECK)
    gold_index_overlap = golden_ids.intersection(index_ids)
    assert len(gold_index_overlap) == 0, f"CRITICAL LEAKAGE: {len(gold_index_overlap)} golden IDs found in retrieval index!"
    
    # 3. Dev vs Retrieval Index disjoint
    dev_index_overlap = dev_ids.intersection(index_ids)
    assert len(dev_index_overlap) == 0, f"Dev IDs found in retrieval index: {dev_index_overlap}"
    
    # 4. Agreement subset must be a true subset of Golden
    assert agreement_ids.issubset(golden_ids), "Agreement subset contains IDs not in Golden Set"
    
    print("ALL LEAKAGE & INTEGRITY CHECKS PASSED: 0% data leakage.")

if __name__ == "__main__":
    test_data_leakage_and_integrity()
