# RoboComposer

Project Structure:
robocomposer/
├── src/
│   ├── __init__.py
│   ├── midi_dataset.py
│   ├── rag_retriever.py
│   ├── music_generator.py
│   ├── llm_orchestrator.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_midi_dataset.py
│   ├── test_rag_retriever.py
│   └── test_llm_orchestrator.py
├── notebooks/
│   └── exploration.ipynb
├── data/
│   ├── raw/          ← MAESTRO goes here
│   └── processed/    ← tokenized output goes here
├── outputs/          ← generated MIDI files
├── .env.example
├── requirements.txt
└── README.md