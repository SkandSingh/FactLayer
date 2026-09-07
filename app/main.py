"""FastAPI application entrypoint.

Starts out with just a health-check route; document upload, fact, and
relationship routes are added as those pieces of the pipeline are built.
"""
from fastapi import FastAPI

from app.routes import documents as documents_routes, facts as facts_routes, relationships as relationships_routes

app = FastAPI(title="FactLayer")

app.include_router(documents_routes.router)
app.include_router(facts_routes.router)
app.include_router(relationships_routes.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
