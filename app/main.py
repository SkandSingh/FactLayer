"""FastAPI application entrypoint.

Starts out with just a health-check route; document upload, fact, and
relationship routes are added as those pieces of the pipeline are built.
"""
from fastapi import FastAPI

app = FastAPI(title="FactLayer")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
