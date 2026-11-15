"""Inject / restore deliberate Kubernetes failures for the 'engineering defense' demo.

  python scripts/failure_k8s.py bad-model-path   # ConfigMap points to a missing model -> readiness 503
  python scripts/failure_k8s.py bad-image        # non-existent image tag -> ImagePullBackOff
  python scripts/failure_k8s.py low-memory       # 20Mi limit -> OOMKilled / CrashLoopBackOff
  python scripts/failure_k8s.py restore          # back to the known-good manifests
"""
from __future__ import annotations

import json
import subprocess
import sys

NS = "mlops-agent"


def kubectl(*args: str) -> None:
    print("$ kubectl", " ".join(args))
    subprocess.run(["kubectl", *args], check=True)


def patch(kind: str, name: str, body: dict) -> None:
    kubectl("-n", NS, "patch", kind, name, "--type", "merge", "-p", json.dumps(body))


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "bad-model-path":
        patch("configmap", "ml-inference-config", {"data": {"MODEL_PATH": "/app/models/does_not_exist.joblib"}})
        kubectl("-n", NS, "rollout", "restart", "deployment/ml-inference")
    elif mode == "bad-image":
        kubectl("-n", NS, "set", "image", "deployment/ml-inference", "inference=mlops-inference:9.9.9")
    elif mode == "low-memory":
        patch("deployment", "ml-inference", {"spec": {"template": {"spec": {"containers": [
            {"name": "inference", "resources": {"limits": {"memory": "20Mi"}, "requests": {"memory": "10Mi"}}}]}}}})
    elif mode == "restore":
        kubectl("apply", "-f", "kubernetes/configmap.yaml", "-f", "kubernetes/deployment.yaml")
        kubectl("-n", NS, "rollout", "restart", "deployment/ml-inference")
        kubectl("-n", NS, "rollout", "status", "deployment/ml-inference", "--timeout=120s")
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
