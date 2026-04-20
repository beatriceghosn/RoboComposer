import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from src.llm_orchestrator import LLMOrchestrator

orchestrator = LLMOrchestrator()

test_prompts = [
    "A sad, slow piece reminiscent of Chopin",
    "An upbeat jazzy tune in a major key",
    "A tense, dramatic cinematic score",
    "A futuristic sounding chase through a rainy forest",
]

print("--- Testing LLMOrchestrator ---\n")

for prompt in test_prompts:
    print(f"Input:  {prompt}")
    try:
        result = orchestrator.parse_user_prompt(prompt)
        print(f"Output: {json.dumps(result, indent=2)}")
        print(f"Valid:  {orchestrator.validate_constraints(result)}")
    except ValueError as e:
        print(f"Error:  {e}")
    print()