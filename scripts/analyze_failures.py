# scripts/analyze_failures.py
import sys
import os
import json
from collections import Counter

sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding="utf-8")

from src.pipeline import get_pipeline

with open("data/golden/golden_set.json", "r", encoding="utf-8") as f:
    gold = json.load(f)

pipe = get_pipeline()

fn_cases = []
fp_cases = []
intent_mismatches = []

for idx, c in enumerate(gold):
    pred = pipe.process_case(c["customer_text"])
    if c["gold_decision"] == "escalate" and pred["decision"] == "auto_handle":
        fn_cases.append({
            "case_id": c["case_id"],
            "text": c["customer_text"],
            "gold_intent": c["gold_intent"],
            "pred_intent": pred["intent"],
            "confidence": pred["confidence"],
            "gold_reason": c["gold_reason"],
            "pred_reason": pred["reason"],
            "rule_triggered": pred["rule_triggered"]
        })
    elif c["gold_decision"] == "auto_handle" and pred["decision"] == "escalate":
        fp_cases.append({
            "case_id": c["case_id"],
            "text": c["customer_text"],
            "gold_intent": c["gold_intent"],
            "pred_intent": pred["intent"],
            "confidence": pred["confidence"],
            "gold_reason": c["gold_reason"],
            "pred_reason": pred["reason"],
            "rule_triggered": pred["rule_triggered"]
        })
        
    if c["gold_intent"] != pred["intent"]:
        intent_mismatches.append({
            "case_id": c["case_id"],
            "text": c["customer_text"],
            "gold_intent": c["gold_intent"],
            "pred_intent": pred["intent"],
            "confidence": pred["confidence"]
        })

print(f"Total Under-escalation FN cases: {len(fn_cases)}")
for i, c in enumerate(fn_cases):
    print(f"\n--- FN CASE #{i+1} [Case ID: {c['case_id']}] ---")
    print(f"Customer: {c['text']}")
    print(f"Gold: {c['gold_intent']} (escalate) -> Pred: {c['pred_intent']} ({c['confidence']:.2f})")
    print(f"System Reason: {c['pred_reason']}")

print(f"\nTotal Over-escalation FP cases: {len(fp_cases)}")
for i, c in enumerate(fp_cases[:5]):
    print(f"\n--- FP CASE #{i+1} [Case ID: {c['case_id']}] ---")
    print(f"Customer: {c['text']}")
    print(f"Gold: {c['gold_intent']} (auto_handle) -> Pred: {c['pred_intent']} ({c['confidence']:.2f})")
    print(f"System Reason: {c['pred_reason']}")

print(f"\nTotal Intent Mismatches: {len(intent_mismatches)}")
pairs = Counter([(m["gold_intent"], m["pred_intent"]) for m in intent_mismatches])
for p, cnt in pairs.most_common(8):
    print(f"  {p[0]:30s} -> {p[1]:30s} : {cnt} cases")
