# scripts/train_classifier.py
import sys
import os
import json
import re
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

sys.stdout.reconfigure(encoding="utf-8")

def train_intent_classifier():
    print("Loading cases and eval IDs...")
    cases_df = pd.read_parquet("data/processed/cases.parquet")
    
    with open("data/dev/dev_set.json", "r", encoding="utf-8") as f:
        dev_cases = json.load(f)
    with open("data/golden/golden_set.json", "r", encoding="utf-8") as f:
        golden_cases = json.load(f)
        
    held_out_ids = {str(c["case_id"]) for c in dev_cases} | {str(c["case_id"]) for c in golden_cases}
    train_pool = cases_df[~cases_df["case_id"].astype(str).isin(held_out_ids)].copy()
    print(f"Clean training pool: {len(train_pool):,} cases.")
    
    intent_vocab = {
        "hardware_damage_repair": ["crack", "cracked", "shatter", "shattered", "broken screen", "broken glass", "water damage", "dropped", "damage", "repair", "dent", "smashed", "swollen"],
        "warranty_applecare": ["applecare", "warranty", "coverage", "repair status", "dispatch", "claim status", "serial number", "covered under"],
        "purchase_billing": ["charged", "charge", "charges", "billing", "bill", "refund", "subscription", "subscriptions", "unauthorized", "itunes bill", "apple.com/bill", "receipt"],
        "account_appleid_icloud": ["apple id", "appleid", "icloud", "password", "passcode", "locked", "disabled", "2fa", "two factor", "verification code", "iforgot", "login", "sign in"],
        "battery_performance": ["battery", "drain", "draining", "drains", "percentage", "overheat", "overheating", "hot", "charging", "charger", "shut down"],
        "software_bug_ios": ["ios", "update", "freeze", "freezing", "freezes", "frozen", "crash", "crashing", "crashes", "glitch", "glitches", "bug", "bugs", "lag", "lagging", "stuck", "keyboard", "bluetooth", "wifi"],
        "device_setup_howto": ["how do i", "how to", "transfer", "setup", "switch", "restore", "move to ios", "airdrop", "backup data", "new phone"],
        "product_availability_preorder": ["stock", "in stock", "preorder", "pre-order", "shipping", "delivery", "arrive", "order status", "store pickup", "trade in"],
        "general_feedback_other": ["thank you", "thanks apple", "worst", "hate", "love", "sucks", "terrible", "genius bar", "kudos", "useless"]
    }
    
    labeled_samples = []
    for idx, row in train_pool.iterrows():
        text = str(row["customer_text"]).lower()
        scores = {}
        for intent, kws in intent_vocab.items():
            sc = sum(1 for kw in kws if kw in text)
            if sc > 0:
                scores[intent] = sc
        if scores:
            sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_intent, best_score = sorted_intents[0]
            second_score = sorted_intents[1][1] if len(sorted_intents) > 1 else 0
            if best_score >= 1 and (best_score - second_score >= 1 or len(sorted_intents) == 1):
                labeled_samples.append({"text": row["customer_text"], "intent": best_intent})
                
    train_df = pd.DataFrame(labeled_samples)
    print(f"Extracted {len(train_df):,} distinct labeled training examples.")
    print(train_df["intent"].value_counts())
    
    # Subsample balanced training set (up to 300 per class)
    balanced_dfs = []
    for intent, grp in train_df.groupby("intent"):
        balanced_dfs.append(grp.sample(n=min(len(grp), 300), random_state=42))
    train_df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    
    vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_df_balanced["text"])
    y_train = train_df_balanced["intent"]
    
    base_clf = LogisticRegression(C=2.5, max_iter=1000, class_weight="balanced")
    calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
    calibrated_clf.fit(X_train, y_train)
    
    # Evaluate on Dev Set
    X_dev = vectorizer.transform([c["customer_text"] for c in dev_cases])
    y_dev = [c["gold_intent"] for c in dev_cases]
    y_dev_pred = calibrated_clf.predict(X_dev)
    
    print("\n--- BALANCED DEV SET VALIDATION REPORT ---")
    print(classification_report(y_dev, y_dev_pred, zero_division=0))
    
    model_bundle = {
        "vectorizer": vectorizer,
        "classifier": calibrated_clf,
        "classes": list(calibrated_clf.classes_),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "trained_samples": len(train_df_balanced)
    }
    
    out_path = "data/processed/intent_classifier.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Saved balanced calibrated classifier to {out_path}.")

if __name__ == "__main__":
    train_intent_classifier()
