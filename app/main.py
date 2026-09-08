"""FastAPI application entrypoint.

Starts out with just a health-check route; document upload, fact, and
relationship routes are added as those pieces of the pipeline are built.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes import documents as documents_routes, facts as facts_routes, relationships as relationships_routes, stats as stats_routes, ui as ui_routes

app = FastAPI(title="FactLayer")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(documents_routes.router)
app.include_router(facts_routes.router)
app.include_router(relationships_routes.router)
app.include_router(stats_routes.router)
app.include_router(ui_routes.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
