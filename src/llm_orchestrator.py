import os
import json
import anthropic

class LLMOrchestrator:
    """
    Uses Claude to translate a user's natural language prompt into
    a structured JSON dict of musical constraints for the generator.
    """

    SYSTEM_PROMPT = """You are a music composition orchestrator.
Given a user's description of the music they want, output ONLY a valid JSON object
with the following keys:
  - mood (str): e.g. "melancholic", "joyful", "tense"
  - tempo (str): "slow", "moderate", or "fast"
  - key (str): e.g. "D minor", "G major"
  - style (str): a composer or style reference, e.g. "Chopin", "Debussy", "Baroque"
  - notes (str): any additional constraints in plain English
Output no other text. Only the JSON object."""

    def __init__(self, model: str = "claude-opus-4-5"):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def parse_user_prompt(self, user_input: str) -> dict:
        """
        Sends user_input to Claude and returns the parsed JSON constraint dict.
        Raises ValueError if the response is not valid JSON.
        """
        raise NotImplementedError

    def _call_claude(self, user_input: str) -> str:
        """Raw API call — returns the string content of Claude's response."""
        raise NotImplementedError

    def validate_constraints(self, constraints: dict) -> bool:
        """Checks that the required keys are present and values are non-empty."""
        required = {"mood", "tempo", "key", "style"}
        return required.issubset(constraints.keys()) and all(constraints[k] for k in required)