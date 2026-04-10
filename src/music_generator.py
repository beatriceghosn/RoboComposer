class MusicGenerator:
    """
    Wraps the fine-tuned MIDI generation model.
    Takes structured constraints + optional RAG context snippets
    and produces a MIDI file.
    """

    def __init__(self, model_path: str = None, device: str = "cpu"):
        """
        Args:
            model_path: Path to fine-tuned model weights. If None, uses base model.
            device: "cpu" or "cuda".
        """
        self.model_path = model_path
        self.device = device
        self.model = None  # TODO: load model here
        raise NotImplementedError

    def generate(
        self,
        constraints: dict,
        context_snippets: list = None,
        max_tokens: int = 512,
        temperature: float = 1.0,
    ) -> bytes:
        """
        Generates a MIDI file given constraints from LLMOrchestrator
        and optional RAG context snippets.
        Returns raw MIDI bytes.
        """
        raise NotImplementedError

    def save_midi(self, midi_bytes: bytes, output_path: str) -> None:
        """Writes generated MIDI bytes to a .mid file."""
        raise NotImplementedError