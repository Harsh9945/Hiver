# src/route.py
import re
from typing import Dict, Any
from src.taxonomy import taxonomy

class EscalationRouter:
    """
    FR-3 Escalation Router for AppleSupport.
    Composes multi-signal decision logic:
    - Sensitive intent override (hardware, billing, warranty)
    - Classifier confidence gate (T_conf = 0.45)
    - Retrieval similarity gate (T_sim = 0.25)
    - Groundedness quality gate (T_ground = 0.35)
    - Customer negative sentiment / churn risk gate
    
    All thresholds calibrated on dedicated dev set (data/dev/dev_set.json).
    """
    def __init__(
        self,
        conf_threshold: float = 0.45,
        sim_threshold: float = 0.25,
        ground_threshold: float = 0.35
    ):
        self.conf_threshold = conf_threshold
        self.sim_threshold = sim_threshold
        self.ground_threshold = ground_threshold
        
        # High-risk / urgent sentiment indicators requiring human intervention
        self.anger_pattern = re.compile(
            r"\b(worst|hate|sucks|terrible|fucking|shit|bullshit|useless|unacceptable|"
            r"furious|lawsuit|ridiculous|steal|robbery|attorney|lawyer|police|fraud)\b", 
            re.I
        )

    def route(
        self,
        customer_text: str,
        intent: str,
        confidence: float,
        max_similarity: float,
        groundedness_score: float
    ) -> Dict[str, Any]:
        """
        Determines auto_handle vs escalate decision and produces human/machine readable reason.
        """
        clean_text = str(customer_text).strip()
        
        # 1. Customer Sentiment & High-Frustration / Legal Risk Gate
        if self.anger_pattern.search(clean_text):
            return {
                "decision": "escalate",
                "reason": "Escalated to senior agent: detected high customer frustration, profanity, or legal churn threat.",
                "rule_triggered": "escalate_customer_sentiment",
                "thresholds": {"sentiment_match": True}
            }

        # 2. Sensitive Intent Policy Override
        if taxonomy.is_sensitive(intent):
            if intent == "hardware_damage_repair":
                reason = "Escalated to human: physical hardware damage requires in-person diagnostic or repair dispatch."
            elif intent == "purchase_billing":
                reason = "Escalated to billing specialist: financial disputes and refund requests require secure human handling."
            elif intent == "warranty_applecare":
                reason = "Escalated to AppleCare specialist: warranty lookup and claim verification require account records."
            else:
                reason = f"Escalated to human: intent '{intent}' is flagged as sensitive under brand support policy."
                
            return {
                "decision": "escalate",
                "reason": reason,
                "rule_triggered": "escalate_sensitive_intent",
                "thresholds": {"sensitive_intent": intent}
            }

        # 3. Classifier Confidence Gate
        if confidence < self.conf_threshold:
            return {
                "decision": "escalate",
                "reason": f"Escalated to human agent: classification confidence ({confidence:.2f}) falls below threshold ({self.conf_threshold:.2f}).",
                "rule_triggered": "escalate_low_confidence",
                "thresholds": {"confidence": confidence, "min_required": self.conf_threshold}
            }

        # 4. Cold-Retrieval / Historical Match Gate
        if max_similarity < self.sim_threshold:
            return {
                "decision": "escalate",
                "reason": f"Escalated to human agent: no similar historical resolution found (retrieval similarity {max_similarity:.2f} < {self.sim_threshold:.2f}).",
                "rule_triggered": "escalate_cold_retrieval",
                "thresholds": {"similarity": max_similarity, "min_required": self.sim_threshold}
            }

        # 5. Reply Groundedness Gate
        if groundedness_score < self.ground_threshold:
            return {
                "decision": "escalate",
                "reason": f"Escalated to human agent: draft reply groundedness ({groundedness_score:.2f}) falls below quality gate ({self.ground_threshold:.2f}).",
                "rule_triggered": "escalate_unreliable_grounding",
                "thresholds": {"groundedness": groundedness_score, "min_required": self.ground_threshold}
            }

        # 6. Auto-Handle Default (Routine Software / How-To / Setup)
        return {
            "decision": "auto_handle",
            "reason": f"Approved for automated reply: verified routine intent '{intent}' with high confidence ({confidence:.2f}) and grounded historical resolution.",
            "rule_triggered": "auto_handle_routine",
            "thresholds": {
                "intent": intent,
                "confidence": confidence,
                "similarity": max_similarity,
                "groundedness": groundedness_score
            }
        }
