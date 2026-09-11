# src/taxonomy.py
import json
import os
from typing import Dict, List, Optional

TAXONOMY_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'taxonomy.json')

class Taxonomy:
    def __init__(self, path: str = TAXONOMY_PATH):
        with open(path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.intents: Dict[str, dict] = self.data['intents']
        self.intent_names: List[str] = list(self.intents.keys())
        self.sensitive_intents: List[str] = [
            k for k, v in self.intents.items() if v.get('sensitive', False)
        ]

    def get_description(self, intent: str) -> str:
        return self.intents.get(intent, {}).get('description', '')

    def get_examples(self, intent: str) -> List[str]:
        return self.intents.get(intent, {}).get('examples', [])

    def is_sensitive(self, intent: str) -> bool:
        return intent in self.sensitive_intents

    def get_default_routing(self, intent: str) -> str:
        return self.intents.get(intent, {}).get('default_routing', 'auto_handle')

    def validate_intent(self, intent: str) -> str:
        if intent in self.intents:
            return intent
        return 'general_feedback_other'

taxonomy = Taxonomy()
