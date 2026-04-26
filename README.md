# RoboComposer

### What it Does
RoboComposer is an interactive, AI-driven music generation platform that translates natural language prompts into piano compositions. The system uses a **Retrieval-Augmented Generation (RAG)** pipeline to search the MAESTRO dataset for MIDI snippets that match the user's desired "vibe." A Google Gemini orchestrator then structures these snippets into a cohesive piece, which is visualized in real-time on a 3D piano interface with synchronized "robot hands."

### Project Structure
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

### Quick Start
1. **Prepare Environment:** Run `pip install -r requirements.txt`.
2. **Set API Key:** Create a `.env` file and add `GEMINI_API_KEY=your_key_here`.
3. **Initialize Memory:** Run `python3 init_index.py` to process the MIDI dataset and build the FAISS index.
4. **Run Server:** Start the backend with `python3 server.py`.
5. **Launch UI:** Open `http://127.0.0.1:8000` in your browser.

### Video Links
* [Demo Video](INSERT_LINK_HERE)
* [Technical Walkthrough](INSERT_LINK_HERE)

### Evaluation

### Individual Contributions
* **Beatrice Ghosn:**
* **Angela Predolac:**