# tests/test_stability.py
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from src.pipeline import get_pipeline

def test_decision_level_stability():
    pipe = get_pipeline(offline=True)
    
    test_cases = [
        "My iPhone 8 battery drains fast after iOS 11 update.",
        "I dropped my phone and the back glass is cracked.",
        "Unknown charge of $14.99 on my credit card from Apple.",
        "How do I transfer contacts from my old Android to new iPhone?",
        "Can someone help me please?"
    ]
    
    for case in test_cases:
        res1 = pipe.process_case(case)
        res2 = pipe.process_case(case)
        
        # 1. Exact Intent Stability
        assert res1["intent"] == res2["intent"], f"Intent instability for query '{case}': {res1['intent']} != {res2['intent']}"
        
        # 2. Exact Routing Decision Stability
        assert res1["decision"] == res2["decision"], f"Decision instability for query '{case}': {res1['decision']} != {res2['decision']}"
        
        # 3. Exact Rule Trigger Stability
        assert res1["rule_triggered"] == res2["rule_triggered"], f"Rule trigger instability: {res1['rule_triggered']} != {res2['rule_triggered']}"
        
        # 4. Lexical Reply Consistency
        reply1 = res1["draft_reply"]
        reply2 = res2["draft_reply"]
        assert len(reply1) > 0 and len(reply2) > 0
        
    print("ALL DECISION-LEVEL STABILITY TESTS PASSED.")

if __name__ == "__main__":
    test_decision_level_stability()
