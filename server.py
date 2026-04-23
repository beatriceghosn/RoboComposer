from io import BytesIO
from pathlib import Path
import traceback

from flask import Flask, jsonify, request, send_file, send_from_directory

from src.llm_orchestrator import LLMOrchestrator
from src.rag_retriever import RAGRetriever
from src.music_generator import MusicGenerator

app = Flask(__name__, static_folder="frontend", static_url_path="")

orchestrator = None
retriever = None
generator = None


def get_orchestrator():
    global orchestrator
    if orchestrator is None:
        orchestrator = LLMOrchestrator()
    return orchestrator


def get_retriever():
    global retriever
    if retriever is None:
        retriever = RAGRetriever(index_path="data/processed/rag_index.faiss")
    return retriever


def get_generator():
    global generator
    if generator is None:
        generator = MusicGenerator()
    return generator


def build_rag_query(constraints: dict) -> str:
    parts = [
        constraints.get("mood", ""),
        constraints.get("tempo", ""),
        constraints.get("key", ""),
        constraints.get("style", ""),
        constraints.get("notes", ""),
    ]
    return " ".join(str(p).strip() for p in parts if str(p).strip())


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/api/compose", methods=["POST"])
def compose():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return jsonify({"error": "Missing prompt."}), 400

    try:
        orch = get_orchestrator()
        ret = get_retriever()
        gen = get_generator()

        constraints = orch.parse_user_prompt(prompt)
        rag_query = build_rag_query(constraints)
        context_snippets = ret.retrieve(rag_query, top_k=3)

        midi_bytes = gen.generate(
            constraints=constraints,
            context_snippets=context_snippets,
            max_tokens=512,
            temperature=0.9,
        )

        return send_file(
            BytesIO(midi_bytes),
            mimetype="audio/midi",
            as_attachment=False,
            download_name="generated.mid",
        )

    except FileNotFoundError as e:
        return jsonify({"error": f"Missing file: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8000)