# tests/test_pipeline.py
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from src.pipeline import get_pipeline

def test_pipeline_pure_function():
    pipe = get_pipeline(offline=True)
    
    # 1. Standard Query
    res = pipe.process_case("My iPhone 8 battery drains very quickly after updating to iOS 11. What can I do?")
    assert isinstance(res, dict)
    for key in ["customer_text", "intent", "confidence", "draft_reply", "groundedness_score", "decision", "reason"]:
        assert key in res, f"Missing key '{key}' in pipeline output"
        
    assert res["decision"] in ["auto_handle", "escalate"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert 0.0 <= res["groundedness_score"] <= 1.0
    assert len(res["draft_reply"]) > 5
    
    # 2. Hardware Damage (Forced Escalate)
    res_hw = pipe.process_case("I dropped my phone on concrete and the front glass screen is completely cracked and shattered.")
    assert res_hw["decision"] == "escalate"
    assert res_hw["intent"] == "hardware_damage_repair"
    assert "hardware" in res_hw["reason"].lower() or "screen" in res_hw["reason"].lower()
    
    # 3. High Frustration / Profanity (Sentiment Escalate)
    res_angry = pipe.process_case("This is the worst customer service ever! You guys are fucking thieves and I will file a lawsuit!")
    assert res_angry["decision"] == "escalate"
    assert "sentiment" in res_angry["rule_triggered"] or "escalate" in res_angry["decision"]
    
    # 4. Edge Cases: Empty text and whitespace
    res_empty = pipe.process_case("   ")
    assert res_empty["decision"] in ["auto_handle", "escalate"]
    
    print("ALL PIPELINE CONTRACT & EDGE-CASE TESTS PASSED.")

if __name__ == "__main__":
    test_pipeline_pure_function()
