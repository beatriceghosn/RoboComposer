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
- Dataset Exploration: EDA on the MAESTRO dataset, exploring size, schema, composer distribution, year distribution, etc. to better understand the data.
- LLM Integration: Architected the llm_orchestrator.py to handle prompt-to-JSON mapping and structured AI responses.
- System Integration: Developed the Flask API backend and the interactive frontend UI, ensuring seamless communication between the Python logic and the Tone.js audio engine.

**Angela Predolac:**

- RAG Pipeline: Engineered the rag_retriever.py modules, implementing FAISS for high-performance semantic search.
- Music Generation: Developed music_generator.py to handle the reconstruction of MIDI files from retrieved motifs and LLM instructions.
- Search Logic: Implemented the core retrieval algorithms to ensure musical coherence between user prompts and retrieved snippets.

### Project Structure

```
RoboComposer/
├── data/
│   └── maestro-v3.0.0
│   └── processed
├── frontend/
│   ├── .gitkeep
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── notebooks/
│   └── exploration.ipynb
├── outputs/
│   ├── full_pipeline_output.mid
│   ├── test_dramatic.mid
│   ├── test_no_rag.mid
│   └── test_with_rag.mid
├── src/
│   ├── __pycache__/
│   ├── .gitkeep
│   ├── llm_orchestrator.py
│   ├── midi_dataset.py
│   ├── music_generator.py
│   ├── rag_retriever.py
│   └── tokenizer.py
├── tests/
│   ├── __pycache__/
│   ├── demo_generate.py
│   ├── test_dataset.py
│   ├── test_llm_orchestrator.py
│   ├── test_midi_dataset.py
│   ├── test_music_generator.py
│   ├── test_rag_retriever.py
│   └── test_tokenizer.py
├── .env
├── .env.example
├── .gitignore
├── attribution.md
├── init_index.py
├── LICENSE
├── maestro-v3.0.0-midi.zip
├── README.md
├── requirements.txt
└── server.py
