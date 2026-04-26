# Attribution of AI-Assisted Development

This project, RoboComposer, was developed with the assistance of AI tools, primarily ChatGPT, as a coding and conceptual support resource. All core design decisions, system architecture, and final implementations were guided and validated by the project authors.

## Summary of AI Usage

AI tools were used in the following ways:

### 1. Code Assistance and Debugging
ChatGPT was used to:
- Debug Python errors such as invalid MIDI note timings, dataset parsing issues, and import errors in midi_dataset.py and tokenizer.py
- Suggest fixes for runtime errors in `music_generator.py` and `rag_retriever.py`
- Provide guidance on structuring and debugging Flask backend routes, including request handling and API responses
- Assist in connecting the frontend UI with backend endpoints, including handling JSON payloads and response formats
- Assist in resolving issues with local development setup (e.g., `.env`, Python environments, package installation)

All suggested code was reviewed, tested, and modified as needed before being integrated into the project.

### 2. Implementation Guidance
ChatGPT was used to:
- Suggest initial scaffolding for components such as:
  - `RAGRetriever` (FAISS-based retrieval system)
  - `MusicGenerator` (baseline MIDI generation logic)
  - `LLMOrchestrator` (prompt-to-JSON constraint parsing)
  -  `midi_dataset.py` and `tokenizer.py` (MIDI preprocessing and tokenization)
- Provide examples of how to structure a multi-stage pipeline (LLM → RAG → generation)
- Help design integration logic between modules

The final implementations were adapted, extended, and debugged by the authors.

### 3. System Design and Architecture
AI assistance was used to:
- Refine the overall system architecture
- Clarify how retrieval-augmented generation (RAG) could be applied to symbolic music generation
- Suggest evaluation strategies (e.g., ablation studies comparing RAG vs. no-RAG)

All architectural decisions were made by the authors based on project requirements and course guidelines.

### 4. Documentation and Writing Support
ChatGPT was used to:
- Help draft and refine explanations for:
  - System components
  - Evaluation methodology
  - Project structure

All written content was reviewed and edited by the authors.

## What Was Not AI-Generated

The following aspects of the project were primarily developed by the authors:

- Overall project concept and design direction
- Integration of all components into a working system
- Debugging and validation of the full pipeline
- Construction and testing of the RAG pipeline over MIDI metadata
- Design of the evaluation strategy and interpretation of results
- Frontend integration and user interaction flow
- Final decisions on implementation trade-offs and system behavior

## Statement of Responsibility

We take full responsibility for the final code, system behavior, and all submitted materials. AI tools were used as assistants, not as replacements for understanding, and all outputs were carefully reviewed, tested, and incorporated by the authors.
