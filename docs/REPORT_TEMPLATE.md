# Technical Report (template)
1. System architecture (paste/refine `architecture.md` diagram)
2. Tool definitions and permissions (table from README + mutating/confirmation rules + namespace/DAG allow-lists)
3. Evidence screenshots (store in `screenshots/`): DVC (`dvc status`, `.dvc` file, git tags), MLflow (runs table, run page with tags),
   Airflow (DAG graph, successful + failed runs), Kubernetes (`kubectl get pods,svc,deploy`, describe, logs)
4. Five tested conversations - paste the agent answer AND the tool trace (`screenshots/evidence/*.json`)
5. Two failure investigations: symptom, evidence, root cause, resolution
6. Known limitations and improvements (see architecture.md)
