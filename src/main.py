from fastapi import FastAPI

app = FastAPI(
    title="Call Center Compliance API",
    version="0.1.0",
    description="FastAPI scaffold for call center compliance workflows.",
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Call Center Compliance API is running."}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
