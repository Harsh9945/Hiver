# run_tests.py
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding="utf-8")

from tests.test_leakage import test_data_leakage_and_integrity
from tests.test_pipeline import test_pipeline_pure_function
from tests.test_stability import test_decision_level_stability

def main():
    print("Running test suite...")
    print("\n--- Test 1: Data Leakage & Integrity ---")
    test_data_leakage_and_integrity()
    print("\n--- Test 2: Pipeline Contract & Edge Cases ---")
    test_pipeline_pure_function()
    print("\n--- Test 3: Decision-Level Stability ---")
    test_decision_level_stability()
    print("\n==========================================")
    print("      ALL TESTS PASSED SUCCESSFULLY!      ")
    print("==========================================")

if __name__ == "__main__":
    main()
