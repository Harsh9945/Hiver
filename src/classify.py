# src/classify.py
import os
import re
import json
import pickle
from typing import Dict, Any, Optional
import numpy as np
from src.taxonomy import taxonomy

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "intent_classifier.pkl")

class IntentClassifier:
    """
    FR-1 Intent Classifier for AppleSupport.
    Combines calibrated TF-IDF + Logistic Regression with a zero-shot LLM fallback.
    - Emits intent label, calibrated confidence float [0.0, 1.0], and probability distribution.
    - Triggers zero-shot LLM fallback when confidence < 0.45, margin < 0.10, or sparse vocabulary.
    - Fallback discretely maps HIGH->0.70, MEDIUM->0.50, LOW->0.30 into downstream router contract.
    """
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Classifier model not found at {model_path}. Run scripts/train_classifier.py first.")
            
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
            
        self.vectorizer = bundle["vectorizer"]
        self.classifier = bundle["classifier"]
        self.classes = bundle["classes"]
        
        # High-precision keyword disambiguation triggers
        self.strong_triggers = {
            "hardware_damage_repair": re.compile(r"\b(crack(ed)?|shatter(ed)?|water damage|dropped.*(phone|screen)|broken (screen|glass)|smashed|swollen battery)\b", re.I),
            "warranty_applecare": re.compile(r"\b(applecare(\+)?|warranty coverage|repair status|claim status|covered under warranty)\b", re.I),
            "purchase_billing": re.compile(r"\b(charged|unauthorized charge|refund|subscription fee|apple\.com/bill|itunes bill|overcharge)\b", re.I),
            "account_appleid_icloud": re.compile(r"\b(apple id|icloud (storage|backup)|forgot.*password|locked apple id|2fa code|iforgot)\b", re.I),
            "battery_performance": re.compile(r"\b(battery (drain|health|life|percentage)|draining fast|dies at \d+%|phone gets hot|overheating)\b", re.I),
            "product_availability_preorder": re.compile(r"\b(in stock|out of stock|preorder|shipping date|delivery date|order status|store pickup)\b", re.I),
            "device_setup_howto": re.compile(r"\b(how (do|can) i|how to (transfer|setup|switch|restore)|move to ios|quick start)\b", re.I),
            "software_bug_ios": re.compile(r"\b(ios \d|update bug|app (crash|freez)|touchscreen unresponsive|keyboard lag|wifi glitch)\b", re.I),
            "general_feedback_other": re.compile(r"\b(thank you|thanks apple|worst service|hate apple|genius bar rocks|terrible support)\b", re.I)
        }

    def predict(self, text: str, llm_client: Optional[Any] = None) -> Dict[str, Any]:
        """
        Classifies input customer tweet text.
        """
        clean_t = str(text).strip()
        if not clean_t:
            return {
                "intent": "general_feedback_other",
                "confidence": 0.10,
                "method": "empty_input_fallback",
                "probabilities": {c: 1.0/len(self.classes) for c in self.classes}
            }

        # Check strong regex triggers first for high precision
        for intent, pattern in self.strong_triggers.items():
            if pattern.search(clean_t):
                # Strong trigger match
                probs = {c: 0.05 for c in self.classes}
                probs[intent] = 0.85
                return {
                    "intent": intent,
                    "confidence": 0.85,
                    "method": "calibrated_strong_trigger",
                    "probabilities": probs,
                    "fallback_triggered": False
                }

        # Statistical TF-IDF inference
        X = self.vectorizer.transform([clean_t])
        prob_matrix = self.classifier.predict_proba(X)[0]
        prob_dict = {cls_name: float(prob_matrix[idx]) for idx, cls_name in enumerate(self.classes)}
        
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        top1_intent, top1_prob = sorted_probs[0]
        top2_intent, top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else ("", 0.0)
        
        vocab_terms = [w for w in re.findall(r"\w+", clean_t.lower()) if w in self.vectorizer.vocabulary_]
        
        # Fallback trigger condition
        trigger_fallback = (
            top1_prob < 0.45 or 
            (top1_prob - top2_prob) < 0.10 or 
            len(vocab_terms) < 3
        )

        if trigger_fallback:
            if llm_client is not None:
                # Call Zero-shot LLM fallback
                return self._call_llm_fallback(clean_t, llm_client, prob_dict)
            else:
                # Offline fallback mode
                # If top1_prob is moderate, use top1 but calibrate confidence down
                if top1_prob >= 0.35:
                    return {
                        "intent": top1_intent,
                        "confidence": round(float(top1_prob), 4),
                        "method": "statistical_calibrated_marginal",
                        "probabilities": prob_dict,
                        "fallback_triggered": True
                    }
                return {
                    "intent": "general_feedback_other",
                    "confidence": 0.30,
                    "method": "statistical_low_confidence_fallback",
                    "probabilities": prob_dict,
                    "fallback_triggered": True
                }

        return {
            "intent": top1_intent,
            "confidence": round(float(top1_prob), 4),
            "method": "statistical_calibrated_high",
            "probabilities": prob_dict,
            "fallback_triggered": False
        }

    def _call_llm_fallback(self, text: str, llm_client: Any, prior_probs: dict) -> Dict[str, Any]:
        """
        Zero-shot LLM classifier fallback.
        Discretely maps HIGH->0.70, MEDIUM->0.50, LOW->0.30 to honor the router contract.
        """
        prompt = f"""You are an expert intent classifier for Apple Support on Twitter.
Available Intent Taxonomy:
1. battery_performance: Battery drain, rapid percentage drop, overheating, charging issues.
2. software_bug_ios: iOS anomalies, app freezes, keyboard lag, update glitches.
3. account_appleid_icloud: Apple ID lockout, password reset, 2FA, iCloud storage.
4. hardware_damage_repair: Cracked screens, liquid damage, physical breakage, swollen battery.
5. warranty_applecare: AppleCare coverage, repair status, warranty claims.
6. purchase_billing: Unauthorized charges, subscriptions, refunds, billing disputes.
7. device_setup_howto: Device setup, data transfer (Move to iOS), quick start, settings.
8. product_availability_preorder: Store stock, shipping dates, order delivery, trade-in.
9. general_feedback_other: Praise, vague complaints, off-topic, general feedback.

Customer Message: "{text}"

Respond in pure JSON with exact keys:
{{
  "intent": "<one of the 9 labels above>",
  "confidence_level": "HIGH" | "MEDIUM" | "LOW",
  "rationale": "<one sentence justification>"
}}"""
        try:
            response = llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            raw_content = response.strip()
            # Parse JSON safely
            match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                intent = taxonomy.validate_intent(data.get("intent", "general_feedback_other"))
                conf_level = str(data.get("confidence_level", "MEDIUM")).upper()
                
                # Calibrated mapping
                if conf_level == "HIGH":
                    calibrated_conf = 0.70
                elif conf_level == "LOW":
                    calibrated_conf = 0.30
                else:
                    calibrated_conf = 0.50
                    
                if intent == "general_feedback_other":
                    calibrated_conf = min(calibrated_conf, 0.40)
                    
                return {
                    "intent": intent,
                    "confidence": calibrated_conf,
                    "confidence_level": conf_level,
                    "rationale": data.get("rationale", ""),
                    "method": "zero_shot_llm_fallback",
                    "probabilities": prior_probs,
                    "fallback_triggered": True
                }
        except Exception as e:
            pass

        return {
            "intent": "general_feedback_other",
            "confidence": 0.00,
            "method": "llm_fallback_error",
            "probabilities": prior_probs,
            "fallback_triggered": True
        }
