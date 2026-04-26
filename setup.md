# RoboComposer Setup Guide

Follow these steps to set up the RoboComposer environment and initialize the AI retrieval engine on your local machine.

## 1. Prerequisites
Ensure you have the following installed:
* **Python 3.9+**
* **pip** (Python package manager)
* A valid **Google Gemini API Key**

---

## 2. Environment Setup

### Clone and Create Virtual Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd RoboComposer

# Create a virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate
```
---

## 3. Install Dependencies
```bash
pip install -r requirements.txt
```
---

## 4. Configuration
Create a .env file and add Gemini Api Key
```bash
touch .env
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
```
---

## 5. Initializing the RAG Index
# Before running the application, you must build the vector database
```
python3 init_index.py
```
---

## 6. Running the Application
### Start the backend
```
python3 server.py
```

### Launch the UI
Open your web browser and go to http://127.0.0.1:8000
Enter a musical prompt and click Generate.