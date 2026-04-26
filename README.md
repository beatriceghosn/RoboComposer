# RoboComposer

### What it Does

RoboComposer is an interactive, AI-driven music generation platform that translates natural language prompts into piano compositions. The system uses a **Retrieval-Augmented Generation (RAG)** pipeline to search the MAESTRO dataset for MIDI snippets that match the user's desired mood, tone, key, etc. A Google Gemini orchestrator then structures these snippets into a cohesive piece, which is visualized in real-time on a piano interface with synchronized "robot hands."

### Quick Start

1. **Prepare Environment:** Run `pip install -r requirements.txt`.
2. **Set API Key:** Create a `.env` file and add `GEMINI_API_KEY=your_key_here`.
3. **Initialize Memory:** Run `python3 init_index.py` to process the MIDI dataset and build the FAISS index.
4. **Run Server:** Start the backend with `python3 server.py`.
5. **Launch UI:** Open `http://127.0.0.1:8000` in your browser.

### Video Links

- [Demo Video](INSERT_LINK_HERE)
- [Technical Walkthrough](INSERT_LINK_HERE)

### Individual Contributions

**Beatrice Ghosn:** 

- Core Logic & Infrastructure: Developed and tested the midi_dataset.py and tokenizer.py modules for MIDI-to-vector processing.
- LLM Integration: Architected the llm_orchestrator.py to handle prompt-to-JSON mapping and structured AI responses.
- System Integration: Developed the Flask API backend and the interactive frontend UI, ensuring seamless communication between the Python logic and the Tone.js audio engine.

**Angela Predolac:**

- RAG Pipeline: Engineered the rag_retriever.py and vector_store.py modules, implementing FAISS for high-performance semantic search.
- Music Generation: Developed music_generator.py to handle the reconstruction of MIDI files from retrieved motifs and LLM instructions.
- Search Logic: Implemented the core retrieval algorithms to ensure musical coherence between user prompts and retrieved snippets.

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

