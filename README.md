# RoboComposer

## Project Structure

```
robocomposer/
├── src/
│   ├── __init__.py
│   ├── midi_dataset.py       # MIDIDataset — loads & tokenizes MAESTRO MIDI files
│   ├── rag_retriever.py      # RAGRetriever — FAISS index of mood-tagged MIDI phrases
│   ├── music_generator.py    # MusicGenerator — LoRA fine-tuned MIDI transformer
│   ├── llm_orchestrator.py   # LLMOrchestrator — Claude parses user prompt → JSON constraints
│   └── utils.py
├── tests/
│   ├── test_midi_dataset.py
│   ├── test_rag_retriever.py
│   └── test_llm_orchestrator.py
├── notebooks/
│   └── exploration.ipynb     # EDA and tokenization experiments
├── data/
│   ├── raw/                  # MAESTRO dataset (not committed)
│   └── processed/            # Tokenized sequences
├── outputs/                  # Generated .mid files
├── .env.example
├── requirements.txt
└── README.md
```