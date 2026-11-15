"""Interactive terminal client:  python -m agent.cli"""
from __future__ import annotations

import json
import sys

from agent.config import Settings
from agent.logging_utils import configure_logging
from agent.orchestrator import Orchestrator, build_llm
from agent.tools import build_registry

HELP = "Commands: /tools  /confirm <id>  /trace on|off  /quit   (anything else is sent to the agent)"


def main() -> int:
    settings = Settings.from_env()
    configure_logging(settings.log_level, settings.log_file or "logs/agent.jsonl")
    orch = Orchestrator(build_registry(settings), build_llm(settings), settings)
    session, show_trace = None, True
    mode = "LLM" if settings.anthropic_api_key else "offline planner"
    print(f"MLOps agent ready ({mode}, {len(orch.registry.names())} tools). {HELP}")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return 0
        if line == "/tools":
            for t in orch.registry.describe():
                print(f"  [{t['module']:<10}] {t['name']}{'  (mutating)' if t['mutating'] else ''}")
            continue
        if line.startswith("/trace"):
            show_trace = line.endswith("on")
            continue
        resp = orch.confirm(line.split(maxsplit=1)[1], session) if line.startswith("/confirm ") else orch.chat(line, session)
        session = resp.session_id
        if show_trace and resp.tool_calls:
            print("\n[tool trace]")
            for t in resp.tool_calls:
                print(f"  {t.tool:<32} ok={t.ok!s:<5} {t.duration_ms:>8.1f} ms {t.error_type or ''}")
        print(f"\nagent> {resp.answer}")
        if resp.pending_actions:
            print("\n[pending]", json.dumps(resp.pending_actions))


if __name__ == "__main__":
    sys.exit(main())
