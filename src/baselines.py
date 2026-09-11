# src/baselines.py
import re
from typing import Dict, Any

class TrivialBaseline:
    """
    Trivial Baseline required by assignment brief:
    - Intent: Always predicts majority class ('software_bug_ios').
    - Reply: Static boilerplate response.
    - Router: Always escalates to human agents.
    """
    def __init__(self):
        self.majority_intent = "software_bug_ios"
        self.static_reply = (
            "Thanks for reaching out to Apple Support! We'd be glad to help. "
            "Please send us a Direct Message with your device model and details so we can assist: "
            "https://twitter.com/messages/compose?recipient_id=AppleSupport"
        )
        
    def classify(self, text: str) -> Dict[str, Any]:
        return {
            "intent": self.majority_intent,
            "confidence": 0.20,
            "method": "majority_class_trivial"
        }
        
    def draft_reply(self, text: str, retrieved_cases: list = None) -> Dict[str, Any]:
        return {
            "draft_reply": self.static_reply,
            "groundedness_score": 0.05,
            "method": "canned_static_boilerplate"
        }
        
    def route(self, text: str, intent_info: dict, reply_info: dict) -> Dict[str, Any]:
        return {
            "decision": "escalate",
            "reason": "Trivial baseline policy: always escalate all inbound customer messages to human queue.",
            "rule_triggered": "always_escalate_trivial"
        }
        
    def process_case(self, text: str) -> Dict[str, Any]:
        intent_info = self.classify(text)
        reply_info = self.draft_reply(text)
        route_info = self.route(text, intent_info, reply_info)
        return {
            "intent": intent_info["intent"],
            "confidence": intent_info["confidence"],
            "draft_reply": reply_info["draft_reply"],
            "groundedness_score": reply_info["groundedness_score"],
            "generation_mode": "canned_static_boilerplate",
            "decision": route_info["decision"],
            "reason": route_info["reason"],
            "baseline_type": "trivial"
        }


class SimpleBaseline:
    """
    Simple Baseline required by assignment brief:
    - Intent: Rule / keyword-based intent classifier.
    - Reply: Intent-keyed generic template without LLM generation or dynamic conditioning.
    - Router: Keyword heuristic (sensitive terms force escalation; else auto_handle).
    """
    def __init__(self):
        self.intent_keywords = {
            "hardware_damage_repair": re.compile(r"\b(crack|shatter|broken|water|drop|swollen|glass|damage)\b", re.I),
            "purchase_billing": re.compile(r"\b(bill|charge|refund|subscription|money|receipt|bank|card)\b", re.I),
            "warranty_applecare": re.compile(r"\b(applecare|warranty|coverage|claim)\b", re.I),
            "battery_performance": re.compile(r"\b(battery|drain|percentage|overheat|hot|charge|shut down)\b", re.I),
            "account_appleid_icloud": re.compile(r"\b(apple id|icloud|password|passcode|lock|2fa|login)\b", re.I),
            "device_setup_howto": re.compile(r"\b(how do i|how to|transfer|setup|switch|move to ios)\b", re.I),
            "product_availability_preorder": re.compile(r"\b(stock|preorder|ship|order|delivery|available)\b", re.I),
            "software_bug_ios": re.compile(r"\b(ios|update|freeze|crash|glitch|bug|lag|stuck|screen)\b", re.I)
        }
        
        self.canned_templates = {
            "hardware_damage_repair": "We're sorry to hear about the damage to your device. We recommend making an appointment at your nearest Apple Store Genius Bar: https://locate.apple.com",
            "purchase_billing": "We understand your concern regarding billing. Please review your purchase history and request refunds securely at https://reportaproblem.apple.com",
            "warranty_applecare": "You can check your AppleCare coverage and repair options online at https://checkcoverage.apple.com with your serial number.",
            "battery_performance": "We'd be glad to help with battery life. Please check Settings > Battery > Battery Health to view maximum capacity and peak performance capability.",
            "account_appleid_icloud": "For Apple ID assistance, please visit https://iforgot.apple.com to verify your identity and reset your security settings.",
            "device_setup_howto": "Here are step-by-step instructions for device setup and data migration: https://support.apple.com/kb/HT201269",
            "product_availability_preorder": "You can check device availability and track your order status directly at https://apple.com/orderstatus or via the Apple Store app.",
            "software_bug_ios": "We want your device running smoothly. Please make sure you are updated to the latest iOS version and try restarting your device.",
            "general_feedback_other": "Thanks for reaching out to Apple Support. Please let us know if there is a specific technical issue we can assist you with."
        }
        
        self.sensitive_escalate_regex = re.compile(
            r"\b(crack|shatter|broken|water damage|bill|charged|refund|unauthorized|applecare|warranty|swollen|fucking|sucks|hate|lawsuit|police)\b",
            re.I
        )

    def classify(self, text: str) -> Dict[str, Any]:
        for intent, pattern in self.intent_keywords.items():
            if pattern.search(text):
                return {
                    "intent": intent,
                    "confidence": 0.60,
                    "method": "keyword_rule_matching"
                }
        return {
            "intent": "general_feedback_other",
            "confidence": 0.35,
            "method": "keyword_rule_fallback"
        }

    def draft_reply(self, text: str, retrieved_cases: list = None) -> Dict[str, Any]:
        intent_info = self.classify(text)
        reply = self.canned_templates.get(intent_info["intent"], self.canned_templates["general_feedback_other"])
        return {
            "draft_reply": reply,
            "groundedness_score": 0.25,
            "method": "canned_intent_template"
        }

    def route(self, text: str, intent_info: dict, reply_info: dict) -> Dict[str, Any]:
        intent = intent_info["intent"]
        if intent in ["hardware_damage_repair", "purchase_billing", "warranty_applecare"]:
            return {
                "decision": "escalate",
                "reason": f"Simple baseline policy: {intent} is inherently sensitive and routed to human agent.",
                "rule_triggered": "sensitive_intent_heuristic"
            }
        if self.sensitive_escalate_regex.search(text):
            return {
                "decision": "escalate",
                "reason": "Simple baseline heuristic: detected sensitive or high-frustration keywords in customer text.",
                "rule_triggered": "keyword_heuristic_escalate"
            }
        return {
            "decision": "auto_handle",
            "reason": "Simple baseline heuristic: routine technical question with no sensitive trigger words detected.",
            "rule_triggered": "default_auto_handle"
        }

    def process_case(self, text: str) -> Dict[str, Any]:
        intent_info = self.classify(text)
        reply_info = self.draft_reply(text)
        route_info = self.route(text, intent_info, reply_info)
        return {
            "intent": intent_info["intent"],
            "confidence": intent_info["confidence"],
            "draft_reply": reply_info["draft_reply"],
            "groundedness_score": reply_info["groundedness_score"],
            "generation_mode": "canned_intent_template",
            "decision": route_info["decision"],
            "reason": route_info["reason"],
            "baseline_type": "simple"
        }
