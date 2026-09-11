# eval/run_eval.py
import os
import sys
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding="utf-8")
import json
import time
import argparse
import numpy as np
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from src.pipeline import SupportAgentPipeline
from src.baselines import TrivialBaseline, SimpleBaseline
from src.llm import QuotaExhaustedError
from eval.metrics import compute_classification_metrics, compute_routing_metrics, compute_reply_metrics
from eval.judge import LLMJudge

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AI Support Agent against Golden Benchmark")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of cases to evaluate (default 50, use 200 for official headline benchmark)")
    parser.add_argument("--golden-file", type=str, default="data/golden/golden_set.json", help="Path to golden set JSON")
    parser.add_argument("--output-file", type=str, default="data/eval_results.json", help="Path to save evaluation results JSON")
    parser.add_argument("--checkpoint-file", type=str, default="data/eval_checkpoint.json", help="Path to store progress checkpoint")
    parser.add_argument("--mode", type=str, choices=["auto", "live", "offline"], default="auto", help="Execution mode: 'live' (strict 100% LLM), 'offline' (deterministic CPU), 'auto' (detect)")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted evaluation from checkpoint by case_id")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers for live LLM calls")
    return parser.parse_args()

def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"ai_preds": {}, "simp_preds": {}, "triv_preds": {}}

def save_checkpoint(checkpoint_file: str, checkpoint_data: Dict[str, Any]):
    os.makedirs(os.path.dirname(checkpoint_file) or ".", exist_ok=True)
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

def run_evaluation(
    sample_size: int, 
    golden_file: str, 
    output_file: str, 
    checkpoint_file: str = "data/eval_checkpoint.json",
    mode: str = "auto",
    resume: bool = False,
    workers: int = 4
):
    print("=" * 80)
    print("   HIVER TAKE-HOME EVALUATION HARNESS: APPLE SUPPORT AGENT")
    print("=" * 80)
    with open(golden_file, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    checkpoint_data = load_checkpoint(checkpoint_file) if resume else {"ai_preds": {}, "simp_preds": {}, "triv_preds": {}}
    if resume and "target_sample_size" in checkpoint_data:
        sample_size = checkpoint_data["target_sample_size"]
        print(f"Resuming evaluation with target sample size: {sample_size} (restored from checkpoint)")

    checkpoint_data["target_sample_size"] = sample_size
    checkpoint_data["golden_file"] = golden_file
    save_checkpoint(checkpoint_file, checkpoint_data)

    eval_cases = all_cases[:sample_size]
    is_full_size = (sample_size >= len(all_cases))
    
    # Initialize systems
    force_offline = (mode == "offline")
    agent_pipeline = SupportAgentPipeline(offline=force_offline)
    trivial_baseline = TrivialBaseline()
    simple_baseline = SimpleBaseline()
    
    if force_offline:
        judge = LLMJudge(llm_client=None)
        judge.is_live_llm = False
        is_live_llm = False
    else:
        judge = LLMJudge(llm_client=agent_pipeline.llm_client)
        is_live_llm = (agent_pipeline.llm_client is not None and 
                       agent_pipeline.llm_client.is_available and 
                       not agent_pipeline.llm_client.circuit_open)
        if mode == "live" and not is_live_llm:
            raise RuntimeError("Live mode requested, but no active LLM client/key is available or circuit breaker is open.")

    mode_name = f"Live LLM API ({agent_pipeline.llm_client.model} / {len(agent_pipeline.llm_client.clients)}-key pool)" if is_live_llm else "Calibrated Offline Engine (Deterministic CPU)"
    strict_live = (mode == "live")
    
    print(f"Evaluation Mode: {mode_name} (Requested Mode: {mode}, Strict Live: {strict_live})")
    print(f"Cases to Evaluate: {len(eval_cases)} / {len(all_cases)}")
    if is_full_size:
        print(">> OFFICIAL HEADLINE BENCHMARK RUN (N = 200) <<")
    else:
        print(f">> FAST VERIFICATION RUN (N = {sample_size}) <<")
        
    y_true_intent = [c["gold_intent"] for c in eval_cases]
    y_true_decision = [c["gold_decision"] for c in eval_cases]
    
    start_time = time.time()
    
    try:
        # =========================================================================
        # 1. AI Agent (Proposed): Classification, Retrieval, Generation, Routing
        # =========================================================================
        print("\n[1/3] Evaluating AI Agent (Proposed)...", flush=True)
        t_ai0 = time.time()
        
        # Check if AI preds already present in checkpoint
        ai_preds_map = checkpoint_data.get("ai_preds", {})
        remaining_cases_ai = [c for c in eval_cases if str(c["case_id"]) not in ai_preds_map]
        
        if remaining_cases_ai:
            print(f"  Inference for {len(remaining_cases_ai)} pending cases (RAG-Conditioned Synthesis)...", flush=True)
            active_llm = agent_pipeline.llm_client if (is_live_llm and not force_offline) else None
            
            # Process in sub-chunks of 10 cases with live progress and continuous checkpointing
            chunk_step = 10
            for ci in range(0, len(remaining_cases_ai), chunk_step):
                t_chk0 = time.time()
                sub_remaining = remaining_cases_ai[ci:ci + chunk_step]
                
                ai_intermediate = []
                for c in sub_remaining:
                    clf = agent_pipeline.classifier.predict(c["customer_text"])
                    ret = agent_pipeline.retriever.retrieve(c["customer_text"], top_k=3)
                    ai_intermediate.append({
                        "case_id": str(c["case_id"]),
                        "customer_text": c["customer_text"],
                        "intent": clf["intent"],
                        "confidence": clf["confidence"],
                        "max_similarity": ret["max_similarity"],
                        "retrieved_cases": ret["top_k_cases"],
                        "gold_intent": c["gold_intent"],
                        "rubric_checklist": c.get("rubric_checklist", [])
                    })

                gen_replies = agent_pipeline.generator.generate_batch(
                    ai_intermediate, 
                    llm_client=active_llm, 
                    batch_size=2, 
                    max_workers=workers
                )

                new_ai_preds = []
                for c_data, g in zip(ai_intermediate, gen_replies):
                    route = agent_pipeline.router.route(
                        c_data["customer_text"],
                        c_data["intent"],
                        c_data["confidence"],
                        c_data["max_similarity"],
                        g["groundedness_score"]
                    )
                    new_ai_preds.append({
                        "case_id": c_data["case_id"],
                        "customer_text": c_data["customer_text"],
                        "gold_intent": c_data["gold_intent"],
                        "rubric_checklist": c_data["rubric_checklist"],
                        "intent": c_data["intent"],
                        "confidence": c_data["confidence"],
                        "draft_reply": g["draft_reply"],
                        "groundedness_score": g["groundedness_score"],
                        "generation_mode": g.get("generation_mode", "unknown"),
                        "max_similarity": c_data["max_similarity"],
                        "decision": route["decision"],
                        "reason": route["reason"]
                    })

                ai_scores = judge.score_batch(new_ai_preds, batch_size=2, max_workers=workers, strict_live=strict_live)
                for p, s in zip(new_ai_preds, ai_scores):
                    p["judge_scores"] = s
                    ai_preds_map[p["case_id"]] = p

                checkpoint_data["ai_preds"] = ai_preds_map
                save_checkpoint(checkpoint_file, checkpoint_data)
                
                pct = (len(ai_preds_map) / len(eval_cases)) * 100
                print(f"  [AI Agent] Progress: {len(ai_preds_map)}/{len(eval_cases)} cases ({pct:.1f}%) [Chunk elapsed: {time.time()-t_chk0:.2f}s]", flush=True)

        ai_preds = [ai_preds_map[str(c["case_id"])] for c in eval_cases]
        print(f"  AI Agent completed in {time.time() - t_ai0:.2f}s ({len(ai_preds)} cases ready).", flush=True)

        # =========================================================================
        # 2. Simple Baseline (Regex + Templates)
        # =========================================================================
        print("\n[2/3] Evaluating Simple Baseline (Regex + Templates)...", flush=True)
        t_sim0 = time.time()
        simp_preds_map = checkpoint_data.get("simp_preds", {})
        remaining_cases_simp = [c for c in eval_cases if str(c["case_id"]) not in simp_preds_map]
        
        if remaining_cases_simp:
            chunk_step = 10
            for ci in range(0, len(remaining_cases_simp), chunk_step):
                t_chk0 = time.time()
                sub_remaining = remaining_cases_simp[ci:ci + chunk_step]
                new_simp_preds = []
                for c in sub_remaining:
                    res = simple_baseline.process_case(c["customer_text"])
                    res["case_id"] = str(c["case_id"])
                    res["customer_text"] = c["customer_text"]
                    res["gold_intent"] = c["gold_intent"]
                    res["rubric_checklist"] = c.get("rubric_checklist", [])
                    new_simp_preds.append(res)

                simp_scores = judge.score_batch(new_simp_preds, batch_size=2, max_workers=workers, strict_live=strict_live)
                for p, s in zip(new_simp_preds, simp_scores):
                    p["judge_scores"] = s
                    simp_preds_map[p["case_id"]] = p

                checkpoint_data["simp_preds"] = simp_preds_map
                save_checkpoint(checkpoint_file, checkpoint_data)
                pct = (len(simp_preds_map) / len(eval_cases)) * 100
                print(f"  [Simple Baseline] Progress: {len(simp_preds_map)}/{len(eval_cases)} cases ({pct:.1f}%) [Chunk elapsed: {time.time()-t_chk0:.2f}s]", flush=True)

        simp_preds = [simp_preds_map[str(c["case_id"])] for c in eval_cases]
        print(f"  Simple Baseline completed in {time.time() - t_sim0:.2f}s ({len(simp_preds)} cases ready).", flush=True)

        # =========================================================================
        # 3. Trivial Baseline (Majority + Always Escalate)
        # =========================================================================
        print("\n[3/3] Evaluating Trivial Baseline (Majority + Always Escalate)...", flush=True)
        t_triv0 = time.time()
        triv_preds_map = checkpoint_data.get("triv_preds", {})
        remaining_cases_triv = [c for c in eval_cases if str(c["case_id"]) not in triv_preds_map]

        if remaining_cases_triv:
            chunk_step = 10
            for ci in range(0, len(remaining_cases_triv), chunk_step):
                t_chk0 = time.time()
                sub_remaining = remaining_cases_triv[ci:ci + chunk_step]
                new_triv_preds = []
                for c in sub_remaining:
                    res = trivial_baseline.process_case(c["customer_text"])
                    res["case_id"] = str(c["case_id"])
                    res["customer_text"] = c["customer_text"]
                    res["gold_intent"] = c["gold_intent"]
                    res["rubric_checklist"] = c.get("rubric_checklist", [])
                    new_triv_preds.append(res)

                triv_scores = judge.score_batch(new_triv_preds, batch_size=2, max_workers=workers, strict_live=strict_live)
                for p, s in zip(new_triv_preds, triv_scores):
                    p["judge_scores"] = s
                    triv_preds_map[p["case_id"]] = p

                checkpoint_data["triv_preds"] = triv_preds_map
                save_checkpoint(checkpoint_file, checkpoint_data)
                pct = (len(triv_preds_map) / len(eval_cases)) * 100
                print(f"  [Trivial Baseline] Progress: {len(triv_preds_map)}/{len(eval_cases)} cases ({pct:.1f}%) [Chunk elapsed: {time.time()-t_chk0:.2f}s]", flush=True)

        triv_preds = [triv_preds_map[str(c["case_id"])] for c in eval_cases]
        print(f"  Trivial Baseline completed in {time.time() - t_triv0:.2f}s ({len(triv_preds)} cases ready).", flush=True)

    except QuotaExhaustedError as qe:
        print("\n" + "=" * 80)
        print(" [QUOTA EXHAUSTED - EVALUATION PAUSED]")
        print("=" * 80)
        print(f"Message: {qe}")
        print(f"Live API quota exhausted. All completed evaluations safely stored in '{checkpoint_file}'.")
        print("No cases were silently degraded or substituted with heuristic scoring.")
        print(f"To resume when quota resets or with an upgraded key, re-run:")
        print(f"  python eval/run_eval.py --sample-size {sample_size} --mode live --resume")
        print("=" * 80)
        return {"status": "paused_quota_exhausted", "checkpoint_file": checkpoint_file}

    # =========================================================================
    # Compute Metrics & Provenance Audit
    # =========================================================================
    systems = {
        "AI Agent (Proposed)": {"preds": ai_preds},
        "Simple Baseline": {"preds": simp_preds},
        "Trivial Baseline": {"preds": triv_preds}
    }

    elapsed_time = time.time() - start_time
    avg_latency = elapsed_time / float(len(eval_cases))

    # Audit provenance across all systems
    total_judge_evals = 0
    live_judge_evals = 0
    fallback_judge_evals = 0
    
    for sys_name, sys_dict in systems.items():
        for p in sys_dict["preds"]:
            total_judge_evals += 1
            j_mode = p.get("judge_scores", {}).get("judge_mode", "unknown")
            if j_mode == "llm_live_api":
                live_judge_evals += 1
            else:
                fallback_judge_evals += 1

    judge_fallback_rate = fallback_judge_evals / float(max(1, total_judge_evals))
    
    ai_gen_live = sum(1 for p in ai_preds if p.get("generation_mode") in ["llm_grounded_generation", "retrieval_conditioned_synthesis"])
    ai_gen_fallback = len(ai_preds) - ai_gen_live
    ai_gen_fallback_rate = ai_gen_fallback / float(max(1, len(ai_preds)))

    # Strict Certification Rule:
    # is_official_headline requires full sample (200), live mode, and exactly 0% fallback
    is_certified_headline = (
        is_full_size and 
        (judge_fallback_rate == 0.0) and 
        (ai_gen_fallback_rate == 0.0) and 
        is_live_llm
    )
    
    if is_certified_headline:
        benchmark_status = "CERTIFIED_OFFICIAL_HEADLINE"
    elif fallback_judge_evals == 0 and is_live_llm:
        benchmark_status = "VERIFIED_LIVE_SUBSET"
    elif force_offline or (live_judge_evals == 0):
        benchmark_status = "CALIBRATED_OFFLINE_BASELINE"
    else:
        benchmark_status = "DEGRADED_PARTIAL_FALLBACK"

    est_input_tokens = len(eval_cases) * 600 if is_live_llm else 0
    est_output_tokens = len(eval_cases) * 120 if is_live_llm else 0
    est_cost_usd = ((est_input_tokens / 1e6) * 0.15 + (est_output_tokens / 1e6) * 0.60) if is_live_llm else 0.0

    print(f"\nTotal evaluation completed in {elapsed_time:.2f}s ({avg_latency:.3f}s per case).", flush=True)
    print(f"Benchmark Status: {benchmark_status} (is_official_headline: {is_certified_headline})", flush=True)
    print(f"Provenance: Judge Live={live_judge_evals}/{total_judge_evals} (Fallback Rate: {judge_fallback_rate*100:.1f}%)")

    results_summary = {
        "metadata": {
            "sample_size": len(eval_cases),
            "is_official_headline": is_certified_headline,
            "status": benchmark_status,
            "mode": mode_name,
            "elapsed_seconds": round(elapsed_time, 2),
            "avg_latency_per_case_sec": round(avg_latency, 3),
            "estimated_cost_usd": round(est_cost_usd, 4),
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        },
        "provenance": {
            "total_cases": len(eval_cases),
            "generation": {
                "ai_agent_live_count": ai_gen_live,
                "ai_agent_fallback_count": ai_gen_fallback,
                "ai_agent_fallback_rate": round(ai_gen_fallback_rate, 4)
            },
            "judge": {
                "total_evaluations": total_judge_evals,
                "live_evaluations": live_judge_evals,
                "fallback_evaluations": fallback_judge_evals,
                "fallback_rate": round(judge_fallback_rate, 4)
            },
            "integrity_certified": is_certified_headline
        },
        "systems": {}
    }

    for name, sys_data in systems.items():
        preds = sys_data["preds"]
        pred_intents = [p["intent"] for p in preds]
        pred_decisions = [p["decision"] for p in preds]
        groundedness = [p.get("groundedness_score", 0.0) for p in preds]
        similarities = [p.get("max_similarity", 0.0) for p in preds]
        judge_scores = [p["judge_scores"]["overall_score"] for p in preds]
        
        clf_metrics = compute_classification_metrics(y_true_intent, pred_intents)
        route_metrics = compute_routing_metrics(y_true_decision, pred_decisions, cost_fn=5.0, cost_fp=1.0)
        reply_metrics = compute_reply_metrics(groundedness, similarities)
        
        results_summary["systems"][name] = {
            "classification": clf_metrics,
            "routing": route_metrics,
            "reply_quality": reply_metrics,
            "judge_quality": {
                "mean_overall": round(float(np.mean(judge_scores)), 2),
                "correctness": round(float(np.mean([p["judge_scores"]["correctness_groundedness"] for p in preds])), 2),
                "tone": round(float(np.mean([p["judge_scores"]["tone_brand_fit"] for p in preds])), 2),
                "completeness": round(float(np.mean([p["judge_scores"]["completeness"] for p in preds])), 2),
                "actionability": round(float(np.mean([p["judge_scores"]["actionability"] for p in preds])), 2)
            }
        }

    # Print Comparison Table
    print("\n" + "=" * 92)
    print(f"        RESULTS COMPARISON TABLE (N = {len(eval_cases)})")
    print(f"        Status: {benchmark_status} | Mode: {mode_name}")
    print("=" * 92)
    print(f"{'Metric':<34} | {'AI Agent (Proposed)':<20} | {'Simple Baseline':<18} | {'Trivial Baseline':<16}")
    print("-" * 92)
    
    ai_c = results_summary["systems"]["AI Agent (Proposed)"]["classification"]
    sim_c = results_summary["systems"]["Simple Baseline"]["classification"]
    triv_c = results_summary["systems"]["Trivial Baseline"]["classification"]
    
    ai_r = results_summary["systems"]["AI Agent (Proposed)"]["routing"]
    sim_r = results_summary["systems"]["Simple Baseline"]["routing"]
    triv_r = results_summary["systems"]["Trivial Baseline"]["routing"]
    
    ai_rep = results_summary["systems"]["AI Agent (Proposed)"]["reply_quality"]
    sim_rep = results_summary["systems"]["Simple Baseline"]["reply_quality"]
    triv_rep = results_summary["systems"]["Trivial Baseline"]["reply_quality"]
    
    ai_j = results_summary["systems"]["AI Agent (Proposed)"]["judge_quality"]
    sim_j = results_summary["systems"]["Simple Baseline"]["judge_quality"]
    triv_j = results_summary["systems"]["Trivial Baseline"]["judge_quality"]
    
    rows = [
        ("Intent Accuracy", f"{ai_c['accuracy']*100:.1f}%", f"{sim_c['accuracy']*100:.1f}%", f"{triv_c['accuracy']*100:.1f}%"),
        ("Intent Macro-F1", f"{ai_c['macro_f1']:.4f}", f"{sim_c['macro_f1']:.4f}", f"{triv_c['macro_f1']:.4f}"),
        ("Routing Precision (Escalate)", f"{ai_r['precision']*100:.1f}%", f"{sim_r['precision']*100:.1f}%", f"{triv_r['precision']*100:.1f}%"),
        ("Routing Recall (Escalate)", f"{ai_r['recall']*100:.1f}%", f"{sim_r['recall']*100:.1f}%", f"{triv_r['recall']*100:.1f}%"),
        ("Routing F1 (Escalate)", f"{ai_r['f1']:.4f}", f"{sim_r['f1']:.4f}", f"{triv_r['f1']:.4f}"),
        ("Under-Escalation Rate (FN)", f"{ai_r['false_negatives']}/{len(eval_cases)} ({ai_r['false_negatives']/len(eval_cases)*100:.1f}%)", f"{sim_r['false_negatives']}/{len(eval_cases)} ({sim_r['false_negatives']/len(eval_cases)*100:.1f}%)", f"{triv_r['false_negatives']}/{len(eval_cases)} (0.0%)"),
        ("Expected Unit Cost (5:1 penalty)", f"{ai_r['expected_unit_cost']:.4f}", f"{sim_r['expected_unit_cost']:.4f}", f"{triv_r['expected_unit_cost']:.4f}"),
        ("Mean Groundedness Score [0-1]", f"{ai_rep['mean_groundedness']:.4f}", f"{sim_rep['mean_groundedness']:.4f}", f"{triv_rep['mean_groundedness']:.4f}"),
        ("Judge Quality [1-5 scale]", f"{ai_j['mean_overall']:.2f} / 5.0", f"{sim_j['mean_overall']:.2f} / 5.0", f"{triv_j['mean_overall']:.2f} / 5.0")
    ]
    
    for metric_name, ai_val, sim_val, triv_val in rows:
        print(f"{metric_name:<34} | {ai_val:<20} | {sim_val:<18} | {triv_val:<16}")
        
    print("=" * 92)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
    print(f"\nDetailed metrics saved to {output_file}.")
    return results_summary

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        sample_size=args.sample_size, 
        golden_file=args.golden_file, 
        output_file=args.output_file, 
        checkpoint_file=args.checkpoint_file,
        mode=args.mode,
        resume=args.resume,
        workers=args.workers
    )

