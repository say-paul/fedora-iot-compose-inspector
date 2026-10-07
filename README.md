# Fedora IoT Compose Inspector

The inspector checks Fedora IoT compose status, openQA, Quay containers, failed
compose logs, and related open `fedora-iot/iot-distro` issues.

Python 3.14+ may display an upstream LangChain/Pydantic compatibility warning during
AI initialization. The inspector leaves that warning visible rather than disabling AI.

## Failure-analysis structure

- `check_fedora_iot.py` contains the inspection workflow: collect logs, decide their
  order, run bounded per-log analyses, synthesize the findings, validate evidence,
  and report possible existing issues.
- `ai_analysis.py` contains AI prompts and AI-specific helpers for per-log analysis,
  final synthesis, structured-output parsing, and issue-report formatting.
- `utils.py` contains generic helpers for extracting error windows, prioritizing a
  requested item, and shortening report text.
- `ai_models.json` selects separate LangChain models and generation settings for
  `log_analysis` and `synthesis`. Set `AI_MODEL_CONFIG` to use another JSON config.

For a failed compose, sources are analyzed in this order: `deliverables.json`,
`pungi.global.log`, affected-architecture `runroot.log` files, the relevant Koji
task, then remaining architecture logs. Each per-log response can reprioritize only
the already collected remaining sources. A final AI call receives the bounded,
structured per-log results rather than all raw logs.
