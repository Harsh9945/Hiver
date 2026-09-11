# eval/judge_agreement.py
import os
import sys
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding="utf-8")
import json
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
from eval.judge import LLMJudge

def compute_judge_human_agreement(agreement_file: str = "data/golden/judge_agreement_subset.json"):
    print(f"Loading 40-case human-scored agreement subset from {agreement_file}...")
    with open(agreement_file, "r", encoding="utf-8") as f:
        cases = json.load(f)
        
    judge = LLMJudge()
    print(f"Running Judge (Mode: {judge.judge_mode if hasattr(judge, 'judge_mode') else ('Live LLM' if judge.is_live_llm else 'Deterministic Rubric')})...")
    
    human_overall = []
    judge_overall = []
    
    human_dims = {"correctness": [], "tone": [], "completeness": [], "actionability": []}
    judge_dims = {"correctness": [], "tone": [], "completeness": [], "actionability": []}
    
    disagreements = []
    
    actual_modes = []
    for idx, c in enumerate(cases):
        text = c["customer_text"]
        ref_reply = c.get("reference_reply", "")
        intent = c["gold_intent"]
        checklist = c.get("rubric_checklist", [])
        
        # Judge scores the reference / agent reply
        j_score = judge.score_reply(text, ref_reply, intent, checklist)
        h_score = c["human_judge_scores"]
        actual_modes.append(j_score.get("judge_mode", "unknown"))
        
        h_ov = h_score["overall_score"]
        j_ov = j_score["overall_score"]
        
        human_overall.append(h_ov)
        judge_overall.append(j_ov)
        
        human_dims["correctness"].append(h_score["correctness_groundedness"])
        human_dims["tone"].append(h_score["tone_brand_fit"])
        human_dims["completeness"].append(h_score["completeness"])
        human_dims["actionability"].append(h_score["actionability"])
        
        judge_dims["correctness"].append(j_score["correctness_groundedness"])
        judge_dims["tone"].append(j_score["tone_brand_fit"])
        judge_dims["completeness"].append(j_score["completeness"])
        judge_dims["actionability"].append(j_score["actionability"])
        
        diff = abs(h_ov - j_ov)
        if diff >= 0.5:
            disagreements.append({
                "case_id": c["case_id"],
                "customer_text": text,
                "reply": ref_reply,
                "human_score": h_ov,
                "judge_score": j_ov,
                "difference": round(diff, 2),
                "human_notes": h_score.get("human_notes", ""),
                "judge_rationale": j_score.get("rationale", "")
            })
            
    # Metrics
    # Discretize for Cohen Kappa (rounded to nearest integer 1-5)
    h_binned = [int(round(s)) for s in human_overall]
    j_binned = [int(round(s)) for s in judge_overall]
    
    kappa_linear = cohen_kappa_score(h_binned, j_binned, weights="linear")
    corr, p_val = pearsonr(human_overall, judge_overall)
    mae = np.mean(np.abs(np.array(human_overall) - np.array(judge_overall)))
    exact_match = sum(1 for h, j in zip(h_binned, j_binned) if h == j) / float(len(h_binned))
    within_one = sum(1 for h, j in zip(h_binned, j_binned) if abs(h - j) <= 1) / float(len(h_binned))
    
    live_count = sum(1 for m in actual_modes if m == "llm_live_api")
    fallback_count = len(actual_modes) - live_count
    fallback_rate = fallback_count / float(max(1, len(actual_modes)))
    if fallback_rate == 0.0:
        overall_mode = "CERTIFIED_LIVE_LLM"
    elif live_count == 0:
        overall_mode = "CALIBRATED_OFFLINE_RUBRIC"
    else:
        overall_mode = f"HYBRID_PARTIAL_FALLBACK ({live_count} live, {fallback_count} fallback)"

    results = {
        "sample_size": len(cases),
        "judge_mode": overall_mode,
        "provenance": {
            "total_cases": len(cases),
            "live_count": live_count,
            "fallback_count": fallback_count,
            "fallback_rate": round(fallback_rate, 4)
        },
        "cohen_kappa_linear": round(float(kappa_linear), 4),
        "pearson_correlation": round(float(corr), 4),
        "mean_absolute_error": round(float(mae), 4),
        "exact_match_ratio": round(float(exact_match), 4),
        "within_1_point_ratio": round(float(within_one), 4),
        "human_mean_overall": round(float(np.mean(human_overall)), 2),
        "judge_mean_overall": round(float(np.mean(judge_overall)), 2),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements
    }
    
    print("\n==================================================")
    print("      JUDGE-VS-HUMAN AGREEMENT BENCHMARK (40 CASES)")
    print("==================================================")
    print(f"Evaluation Sample Size:      {len(cases)} cases")
    print(f"Judge Mode:                  {results['judge_mode']}")
    print(f"Cohen's Weighted Kappa:      {results['cohen_kappa_linear']:.4f}")
    print(f"Pearson Correlation (r):     {results['pearson_correlation']:.4f}")
    print(f"Mean Absolute Error (MAE):   {results['mean_absolute_error']:.4f}")
    print(f"Exact Agreement (binned):    {results['exact_match_ratio']*100:.1f}%")
    print(f"Within-1-Point Agreement:    {results['within_1_point_ratio']*100:.1f}%")
    print(f"Human Average Score:         {results['human_mean_overall']:.2f} / 5.0")
    print(f"Judge Average Score:         {results['judge_mean_overall']:.2f} / 5.0")
    print("==================================================")
    
    if disagreements:
        print(f"\nTop {min(3, len(disagreements))} Disagreement Cases (for Failure Analysis):")
        for idx, d in enumerate(disagreements[:3]):
            print(f"\nDisagreement #{idx+1} [Case ID: {d['case_id']}]:")
            print(f"  Customer: {d['customer_text']}")
            print(f"  Reply:    {d['reply']}")
            print(f"  Human: {d['human_score']} vs Judge: {d['judge_score']} (Diff: {d['difference']})")
            print(f"  Judge Rationale: {d['judge_rationale']}")
            
    out_file = "data/judge_agreement_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved agreement results to {out_file}.")
    return results

if __name__ == "__main__":
    compute_judge_human_agreement()
