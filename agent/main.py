"""FastAPI interface for the agent: validation in, evidence-based response out.

Run:  uvicorn agent.main:app --port 8001
"""
from __future__ import annotations

import hmac
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from agent.config import Settings
from agent.logging_utils import configure_logging, get_logger, request_id_var
from agent.orchestrator import Orchestrator, build_llm
from agent.schemas import AgentResponse, ChatRequest, ConfirmRequest
from agent.tools import build_registry

log = get_logger("api")


def create_app(orchestrator: Orchestrator | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level, settings.log_file)
    orch = orchestrator or Orchestrator(build_registry(settings), build_llm(settings), settings)
    app = FastAPI(title="MLOps AI Agent", version="1.0.0")

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request_id_var.set(rid)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    def auth(authorization: str | None = Header(default=None)) -> None:
        if settings.agent_api_token:
            supplied = (authorization or "").removeprefix("Bearer ").strip()
            if not hmac.compare_digest(supplied, settings.agent_api_token):
                raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "tools": len(orch.registry.names())}

    @app.get("/tools", dependencies=[Depends(auth)])
    def tools() -> list[dict]:
        return orch.registry.describe()

    @app.post("/chat", response_model=AgentResponse, dependencies=[Depends(auth)])
    def chat(req: ChatRequest) -> AgentResponse:
        return orch.chat(req.message, req.session_id, request_id_var.get())

    @app.post("/confirm", response_model=AgentResponse, dependencies=[Depends(auth)])
    def confirm(req: ConfirmRequest) -> AgentResponse:
        return orch.confirm(req.pending_id, req.session_id, request_id_var.get())

    return app


app = create_app()
