# scripts/tune_dev_router.py
import sys
import json
from src.classify import IntentClassifier
from src.retrieve import Retriever
from src.generate import ReplyGenerator
from src.route import EscalationRouter

sys.stdout.reconfigure(encoding="utf-8")

def evaluate_dev_router():
    print("Loading dev set (35 cases)...")
    with open("data/dev/dev_set.json", "r", encoding="utf-8") as f:
        dev_cases = json.load(f)
        
    classifier = IntentClassifier()
    retriever = Retriever()
    generator = ReplyGenerator()
    router = EscalationRouter(conf_threshold=0.45, sim_threshold=0.25, ground_threshold=0.35)
    
    y_true = []
    y_pred = []
    
    tp = 0 # True positive: gold escalate, pred escalate
    fp = 0 # False positive: gold auto_handle, pred escalate
    tn = 0 # True negative: gold auto_handle, pred auto_handle
    fn = 0 # False negative: gold escalate, pred auto_handle (SEVERE COST)
    
    print("\n--- DEV SET CASE-BY-CASE EVALUATION ---")
    for idx, c in enumerate(dev_cases):
        text = c["customer_text"]
        gold_dec = c["gold_decision"]
        
        # 1. Classify
        clf_res = classifier.predict(text)
        # 2. Retrieve
        ret_res = retriever.retrieve(text, top_k=3)
        # 3. Generate
        gen_res = generator.generate_reply(text, clf_res["intent"], ret_res["top_k_cases"])
        # 4. Route
        route_res = router.route(
            text, 
            clf_res["intent"], 
            clf_res["confidence"], 
            ret_res["max_similarity"], 
            gen_res["groundedness_score"]
        )
        
        pred_dec = route_res["decision"]
        y_true.append(gold_dec)
        y_pred.append(pred_dec)
        
        if gold_dec == "escalate" and pred_dec == "escalate":
            tp += 1
        elif gold_dec == "auto_handle" and pred_dec == "escalate":
            fp += 1
        elif gold_dec == "auto_handle" and pred_dec == "auto_handle":
            tn += 1
        elif gold_dec == "escalate" and pred_dec == "auto_handle":
            fn += 1
            print(f"  [UNDER-ESCALATION FN #{fn}] Case ID: {c['case_id']}")
            print(f"    Text: {text}")
            print(f"    Gold Reason: {c['gold_reason']}")
            print(f"    Pred Reason: {route_res['reason']}")
            
    total = len(dev_cases)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 5:1 Cost Asymmetry Metric
    cost_fn = 5.0
    cost_fp = 1.0
    expected_cost = (cost_fn * fn + cost_fp * fp) / float(total)
    
    # Baseline Always Escalate Cost (FN = 0, FP = gold auto_handle count)
    gold_auto_count = sum(1 for d in y_true if d == "auto_handle")
    baseline_always_cost = (cost_fp * gold_auto_count) / float(total)
    
    print("\n==========================================")
    print("       DEV SET ROUTING BENCHMARK          ")
    print("==========================================")
    print(f"Total Cases:       {total}")
    print(f"True Escalations:  {tp}")
    print(f"False Escalations: {fp}")
    print(f"True Auto-Handles: {tn}")
    print(f"False Auto-Handles (FN - Severe): {fn}")
    print(f"Routing Precision: {precision:.4f}")
    print(f"Routing Recall:    {recall:.4f}")
    print(f"Routing F1-Score:  {f1:.4f}")
    print(f"Expected Unit Cost (5:1):     {expected_cost:.4f}")
    print(f"Always-Escalate Cost (5:1):   {baseline_always_cost:.4f}")
    print("==========================================")

if __name__ == "__main__":
    evaluate_dev_router()
