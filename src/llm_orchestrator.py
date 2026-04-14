import os
import json
# import anthropic
from google import genai
from dotenv import load_dotenv

load_dotenv()

class LLMOrchestrator:
    """
    Uses Gemini to translate a user's natural language prompt into
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

    def __init__(self, model_id: str = "gemini-2.5-flash"):
        # self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        self.model_id = model_id

    def parse_user_prompt(self, user_input: str) -> dict:
        """
        Sends user_input to Gemini and returns the parsed JSON constraint dict.
        Raises ValueError if the response is not valid JSON.
        """
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=user_input,
            config={
                "system_instruction": self.SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        )

        constraints = json.loads(response.text)
        if self.validate_constraints(constraints):
            return constraints
        else:
            raise ValueError("Gemini response error")

    def validate_constraints(self, constraints: dict) -> bool:
        """Checks that the required keys are present and values are non-empty."""
        required = {"mood", "tempo", "key", "style"}
        return required.issubset(constraints.keys()) and all(constraints[k] for k in required)

if __name__ == "__main__":
    orchestrator = LLMOrchestrator()
    try:
        sample_input = "A futuristic sounding chase through a rainy forest"
        result = orchestrator.parse_user_prompt(sample_input)
        print(json.dumps(result, indent=4))
    except Exception as e:
        print(f"Error: {e}")