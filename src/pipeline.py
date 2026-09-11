# src/pipeline.py
import os
import sys
from typing import Dict, Any, Optional

from src.classify import IntentClassifier
from src.retrieve import Retriever
from src.generate import ReplyGenerator
from src.route import EscalationRouter
from src.llm import LLMClient

class SupportAgentPipeline:
    """
    Production AI Support Agent for AppleSupport.
    Provides a pure-function inference interface:
      case_in -> {intent, confidence, draft_reply, groundedness_score, decision, reason}
    """
    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[ReplyGenerator] = None,
        router: Optional[EscalationRouter] = None,
        llm_client: Optional[LLMClient] = None,
        offline: bool = False
    ):
        self.offline = offline
        self.classifier = classifier or IntentClassifier()
        self.retriever = retriever or Retriever()
        self.generator = generator or ReplyGenerator()
        self.router = router or EscalationRouter()
        self.llm_client = None if offline else (llm_client or LLMClient())

    def process_case(self, customer_text: str) -> Dict[str, Any]:
        """
        Pure function processing a single incoming customer tweet.
        """
        clean_text = str(customer_text).strip()
        
        # 1. Intent Classification
        active_llm = None if (self.offline or not self.llm_client) else (self.llm_client if self.llm_client.is_available else None)
        clf_res = self.classifier.predict(clean_text, llm_client=active_llm)
        intent = clf_res["intent"]
        confidence = clf_res["confidence"]
        
        # 2. Historical Retrieval
        ret_res = self.retriever.retrieve(clean_text, top_k=3)
        retrieved_cases = ret_res["top_k_cases"]
        max_similarity = ret_res["max_similarity"]
        
        # 3. Grounded Reply Drafting
        gen_res = self.generator.generate_reply(
            clean_text, 
            intent, 
            retrieved_cases, 
            llm_client=active_llm
        )
        draft_reply = gen_res["draft_reply"]
        groundedness_score = gen_res["groundedness_score"]
        
        # 4. Multi-Signal Escalation Routing
        route_res = self.router.route(
            clean_text,
            intent,
            confidence,
            max_similarity,
            groundedness_score
        )
        decision = route_res["decision"]
        reason = route_res["reason"]
        
        return {
            "customer_text": clean_text,
            "intent": intent,
            "confidence": confidence,
            "classification_method": clf_res.get("method", "calibrated"),
            "draft_reply": draft_reply,
            "groundedness_score": groundedness_score,
            "generation_mode": gen_res.get("generation_mode", "unknown"),
            "decision": decision,
            "reason": reason,
            "rule_triggered": route_res.get("rule_triggered", "unknown"),
            "max_similarity": max_similarity,
            "retrieved_cases": retrieved_cases
        }

# Global singleton instance for clean CLI and eval harness import
default_pipeline = None

def get_pipeline(offline: bool = False) -> SupportAgentPipeline:
    global default_pipeline
    if offline:
        return SupportAgentPipeline(offline=True)
    if default_pipeline is None:
        default_pipeline = SupportAgentPipeline()
    return default_pipeline

def process_case(text: str) -> Dict[str, Any]:
    return get_pipeline().process_case(text)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "My iPhone 7 battery percentage drops from 80% to 10% in half an hour. What should I do?"
        
    res = process_case(query)
    import json
    print(json.dumps(res, indent=2, ensure_ascii=False))
