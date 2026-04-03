# Call Center Compliance API

A minimal FastAPI project scaffold for call center compliance workflows.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` if needed.
4. Run the app:
   ```bash
   uvicorn src.main:app --reload
   ```

## Project Structure

- `src/main.py` - FastAPI application entry point
- `src/utils.py` - Shared utility placeholders
- `src/stt.py` - Speech-to-text placeholders
- `src/nlp.py` - NLP processing placeholders
