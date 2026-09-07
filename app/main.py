"""FastAPI application entrypoint.

Starts out with just a health-check route; document upload, fact, and
relationship routes are added as those pieces of the pipeline are built.
"""
from fastapi import FastAPI

from app.routes import documents as documents_routes

app = FastAPI(title="FactLayer")

app.include_router(documents_routes.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
