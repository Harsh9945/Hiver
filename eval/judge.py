# eval/judge.py
import re
import json
from typing import Dict, Any, Optional
from src.llm import LLMClient

class LLMJudge:
    """
    LLM-as-Judge for AppleSupport reply quality evaluation.
    Evaluates 4 explicit dimensions on a 1-5 integer scale:
    1. Correctness & Groundedness: Technical accuracy & consistency with Apple guidelines.
    2. Tone & Brand Fit: Empathy, politeness, Apple Genius Bar voice.
    3. Completeness: Directly addresses the customer query points vs non-diagnostic deflection.
    4. Actionability: Provides concrete diagnostic next steps, Settings pathways, or official links.
    
    Principled Rubric: The judge prompt uniformly instructs the LLM to penalize non-answers
    and generic deflections that provide no diagnostic troubleshooting.
    """
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        self.is_live_llm = self.llm_client.is_available
        self.judge_mode = "llm_live_api" if self.is_live_llm else "calibrated_rubric_baseline"

    def score_reply(
        self,
        customer_text: str,
        draft_reply: str,
        gold_intent: str,
        rubric_checklist: list = None,
        retrieved_context: list = None
    ) -> Dict[str, Any]:
        """
        Evaluates a draft reply using genuine LLM prompt (when API key is present) or fallback rubric.
        """
        checklist_str = "\n".join([f"- {item}" for item in (rubric_checklist or [])])
        
        if self.is_live_llm:
            prompt = f"""You are a rigorous Apple Support Quality Auditor evaluating an AI Agent's tweet reply to a customer.
Customer Query: "{customer_text}"
Expected Intent: {gold_intent}
Rubric Verification Checklist:
{checklist_str}

Candidate Reply to Evaluate: "{draft_reply}"

UNIFORM RUBRIC CRITERIA (1 to 5 integer scale):
1. correctness_groundedness:
   - 1: Factually wrong, misleading, or hallucinated.
   - 2: Generic deflection with zero technical or diagnostic content.
   - 3: Partially accurate but missing key technical details.
   - 4: Technically sound and consistent with Apple guidelines.
   - 5: Flawless technical diagnosis and alignment with official Apple support protocol.
2. tone_brand_fit:
   - 1: Rude, dismissive, or robotic.
   - 3: Neutral and passable.
   - 5: Highly empathetic, polite, and representative of Apple's Genius Bar voice.
3. completeness:
   - 1: Completely ignores the customer's problem or merely deflects without answering.
   - 2: Mentions the issue but provides no diagnostic explanation or assistance.
   - 3: Addresses part of the customer's inquiry.
   - 4: Addresses all major aspects of the problem.
   - 5: Fully and thoroughly addresses all aspects of the customer query.
4. actionability:
   - 1: Dead end; gives the customer nothing actionable to do.
   - 2: Vague deflection (e.g. only says 'DM us' with no troubleshooting steps).
   - 3: Suggests a general direction but lacks specificity.
   - 4: Directs to a specific tool, setting, or link.
   - 5: Clear, step-by-step diagnostic instructions (e.g. Settings > Battery > Battery Health).

Respond ONLY with a valid JSON object with exact keys:
{{
  "correctness_groundedness": <1-5>,
  "tone_brand_fit": <1-5>,
  "completeness": <1-5>,
  "actionability": <1-5>,
  "overall_score": <1.0-5.0 float average>,
  "rationale": "<2-sentence explanation of scores>"
}}"""
            import time
        from src.llm import QuotaExhaustedError
        if self.is_live_llm and self.llm_client and not self.llm_client.circuit_open:
            try:
                raw_resp = self.llm_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                match = re.search(r"\{.*\}", raw_resp, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    c = int(data.get("correctness_groundedness", 3))
                    t = int(data.get("tone_brand_fit", 4))
                    comp = int(data.get("completeness", 3))
                    a = int(data.get("actionability", 3))
                    # Clamp strictly to [1, 5]
                    c = max(1, min(5, c))
                    t = max(1, min(5, t))
                    comp = max(1, min(5, comp))
                    a = max(1, min(5, a))
                    overall = round((c + t + comp + a) / 4.0, 2)
                    return {
                        "correctness_groundedness": c,
                        "tone_brand_fit": t,
                        "completeness": comp,
                        "actionability": a,
                        "overall_score": overall,
                        "rationale": data.get("rationale", ""),
                        "judge_mode": "llm_live_api"
                    }
            except QuotaExhaustedError:
                pass
            except Exception as e:
                pass

        # Offline heuristic rubric (used ONLY if no LLM API is available)
        reply_lower = draft_reply.lower()
        matched_checklist = 0
        checklist = rubric_checklist or []
        for item in checklist:
            item_kws = [w for w in re.findall(r"\w+", item.lower()) if len(w) > 3]
            if any(kw in reply_lower for kw in item_kws):
                matched_checklist += 1
        checklist_ratio = matched_checklist / max(1, len(checklist))

        # Check if generic deflection
        is_deflection = "please send us a direct message" in reply_lower or "dm us for assistance" in reply_lower
        
        if is_deflection and checklist_ratio == 0:
            c_score, t_score, comp_score, a_score = 2, 4, 1, 2
            rationale = "Offline baseline rubric: generic deflection with zero diagnostic checklist alignment."
        else:
            c_score = 5 if checklist_ratio >= 0.5 else (4 if any(kw in reply_lower for kw in ["settings", "apple.com", "ios"]) else 3)
            t_score = 5 if any(w in reply_lower for w in ["happy to help", "we'd be glad", "here for you"]) else 4
            comp_score = 5 if checklist_ratio >= 0.5 else (4 if len(draft_reply.split()) > 10 else 3)
            a_score = 5 if ("http" in reply_lower or "settings >" in reply_lower) else 3
            rationale = f"Offline baseline rubric: matched {matched_checklist}/{len(checklist)} checklist items."

        overall = round((c_score + t_score + comp_score + a_score) / 4.0, 2)
        return {
            "correctness_groundedness": c_score,
            "tone_brand_fit": t_score,
            "completeness": comp_score,
            "actionability": a_score,
            "overall_score": overall,
            "rationale": rationale,
            "judge_mode": "calibrated_rubric_baseline"
        }

    def score_batch(self, case_items: list, batch_size: int = 2, max_workers: int = 2, strict_live: bool = False) -> list:
        """
        High-throughput batched LLM-as-Judge evaluation.
        Evaluates 2 cases per call to keep token volume safely under free-tier ceilings (~200 tokens/call).
        Eliminates cascading retry loops and supports strict_live checkpoint-and-pause semantics.
        """
        from src.llm import QuotaExhaustedError
        results = [None] * len(case_items)
        if not self.is_live_llm or (self.llm_client and self.llm_client.circuit_open):
            if strict_live:
                raise QuotaExhaustedError("Live LLM is unavailable or circuit breaker is open during strict-live evaluation.")
            for idx, c in enumerate(case_items):
                results[idx] = self.score_reply(
                    c["customer_text"], c["draft_reply"], c["gold_intent"], c.get("rubric_checklist", [])
                )
            return results

        chunks = []
        for i in range(0, len(case_items), batch_size):
            chunks.append((i, case_items[i:i + batch_size]))

        def process_chunk(chunk_data):
            start_idx, sub_cases = chunk_data
            prompt = """You are a rigorous Apple Support Quality Auditor evaluating candidate tweet replies against Apple support guidelines.
Rate each case on a 1-5 integer scale across 4 dimensions:
1. correctness (1: wrong, 2: generic deflection, 4: sound, 5: flawless)
2. tone (1: rude, 3: neutral, 5: Genius Bar voice)
3. completeness (1: ignores issue, 3: partial, 5: thorough)
4. actionability (1: dead end, 2: vague DM, 4: specific tool/setting/link, 5: diagnostic)

Cases to evaluate:
"""
            for j, sc in enumerate(sub_cases):
                chk = "\n".join([f"    * {it}" for it in sc.get("rubric_checklist", [])])
                prompt += f"""
[Case {j+1}]
Query: "{sc['customer_text']}"
Expected Intent: {sc['gold_intent']}
Checklist:
{chk}
Reply: "{sc['draft_reply']}"
"""
            prompt += f"""
Respond ONLY with a valid JSON array of {len(sub_cases)} objects with exact keys:
[
  {{
    "case_index": 1,
    "correctness": <1-5>,
    "tone": <1-5>,
    "completeness": <1-5>,
    "actionability": <1-5>,
    "overall_score": <1.0-5.0 float average>,
    "reason": "<15 words max>"
  }},
  ...
]"""
            try:
                # 2 cases require ~180-220 tokens, safely under Groq's 1000 OTPM limit
                req_tokens = min(350, max(180, 85 * len(sub_cases)))
                raw_resp = self.llm_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=req_tokens
                )
                match = re.search(r"\[.*\]", raw_resp, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                        if len(parsed) == len(sub_cases):
                            chunk_res = []
                            for item in parsed:
                                c = max(1, min(5, int(item.get("correctness", item.get("correctness_groundedness", 3)))))
                                t = max(1, min(5, int(item.get("tone", item.get("tone_brand_fit", 4)))))
                                comp = max(1, min(5, int(item.get("completeness", 3))))
                                a = max(1, min(5, int(item.get("actionability", 3))))
                                ov = float(item.get("overall_score", round((c + t + comp + a) / 4.0, 2)))
                                r_text = item.get("reason", item.get("rationale", ""))
                                chunk_res.append({
                                    "correctness_groundedness": c,
                                    "tone_brand_fit": t,
                                    "completeness": comp,
                                    "actionability": a,
                                    "overall_score": ov,
                                    "rationale": r_text,
                                    "judge_mode": "llm_live_api"
                                })
                            return start_idx, chunk_res
                        else:
                            print(f"  [Judge Parse] Length mismatch: got {len(parsed)}, expected {len(sub_cases)}", flush=True)
                    except Exception as je:
                        print(f"  [Judge Parse Error] {je}\n  Matched string truncated: {match.group(0)[:300]}", flush=True)
                else:
                    print(f"  [Judge Match Error] No JSON array found in raw_resp:\n{raw_resp[:300]}", flush=True)
            except QuotaExhaustedError:
                if strict_live:
                    raise
            except Exception as e:
                print(f"  [Judge Chunk Exception] {type(e).__name__}: {e}", flush=True)
                if 'raw_resp' in locals():
                    print(f"  [Judge Raw Output Truncated] {raw_resp[:200]}...", flush=True)

            if strict_live:
                raise QuotaExhaustedError("Strict live judge failed to process chunk via LLM API.")

            # Immediate calibrated heuristic fallback for this chunk without 10x retry cascade
            single_res = []
            for sc in sub_cases:
                single_res.append(self.score_reply(
                    sc["customer_text"], sc["draft_reply"], sc["gold_intent"], sc.get("rubric_checklist", [])
                ))
            return start_idx, single_res

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_results = list(executor.map(process_chunk, chunks))

        for start_idx, chunk_scores in chunk_results:
            for offset, score in enumerate(chunk_scores):
                results[start_idx + offset] = score

        return results
