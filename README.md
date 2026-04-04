# Call Center Compliance API

A minimal FastAPI project scaffold for call center compliance workflows.

## Environment Variables

Create a `.env` file in the project root:

- `API_KEY` - required API key expected in `x-api-key` header
- `LOG_LEVEL` - logging level (`INFO` by default)
- `MAX_REQUEST_BYTES` - max HTTP request size in bytes (default `15728640`)
- `MAX_AUDIO_BASE64_CHARS` - max Base64 string length (default `10000000`)
- `MAX_AUDIO_BYTES` - max decoded audio size in bytes (default `8388608`)
- `REQUEST_TIMEOUT_SECONDS` - max processing time per request (default `180`)
- `TEMP_AUDIO_DIR` - temp decoded audio folder (default `temp_audio`)

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

If you created or changed `.env` while server is running, restart Uvicorn.

## API Authentication

`/api/call-analytics` is protected using header API key auth:

- Header: `x-api-key: <your_api_key>`
- In Swagger UI, click **Authorize** and provide the API key value.

## Notes for Stability

- Requests larger than configured limits return `413 Payload Too Large`.
- Each request receives `x-request-id` and `x-process-time-ms` in response headers.
- Unhandled errors are converted to structured JSON responses to avoid crash loops.

## Troubleshooting

- Browser warning `Tracking Prevention blocked access to storage` is from browser privacy policy and does not break API execution.
- Swagger warning about deep-link whitespace is from Swagger UI internals and is non-blocking.
- If `/api/call-analytics` returns `503 Authentication is not configured on server`, set `API_KEY` in `.env` and restart server.
- Use the same value in Swagger **Authorize** as `x-api-key`.

## Project Structure

- `src/main.py` - FastAPI application entry point
- `src/utils.py` - Shared utility placeholders
- `src/stt.py` - Speech-to-text placeholders
- `src/nlp.py` - NLP processing placeholders
