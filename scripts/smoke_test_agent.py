"""Send the 5 demo scenarios to a running agent API and save the JSON evidence to screenshots/evidence/.

  uvicorn agent.main:app --port 8001   (in another terminal)
  python scripts/smoke_test_agent.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

BASE = os.getenv("AGENT_URL", "http://localhost:8001")
HEADERS = {"Authorization": f"Bearer {os.getenv('AGENT_API_TOKEN')}"} if os.getenv("AGENT_API_TOKEN") else {}
SCENARIOS = {
    "1_dataset_provenance": "Which dataset version was used to train the best model by f1? How can I reproduce it?",
    "2_model_selection": "Compare the last five training runs and recommend the best candidate based on F1 score.",
    "3_pipeline_execution": "Trigger the model training pipeline and report whether execution succeeded.",
    "4_deployment_diagnosis": "Why is my prediction API unavailable?",
    "5_multi_tool": "Find the best recent model, identify the dataset version used to train it, check the deployment "
                    "status, and explain whether retraining is necessary.",
}


def main() -> None:
    out = Path("screenshots/evidence")
    out.mkdir(parents=True, exist_ok=True)
    for name, msg in SCENARIOS.items():
        r = requests.post(f"{BASE}/chat", json={"message": msg}, headers=HEADERS, timeout=180).json()
        (out / f"{name}.json").write_text(json.dumps(r, indent=2))
        print(f"\n=== {name} ===\n{r.get('answer', r)}\n tools: {[t['tool'] for t in r.get('tool_calls', [])]}")
        for p in r.get("pending_actions", []):
            print(f"  pending confirmation: {p['pending_id']}  (POST /confirm to execute)")


if __name__ == "__main__":
    main()
