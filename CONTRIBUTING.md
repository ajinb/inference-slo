# Contributing

Thanks for your interest in inference-slo.

- Run `pip install -e ".[dev]"`, then `pytest -q` and `ruff check src tests examples`.
- **The simulator is deterministic.** Every source of randomness takes an explicit seed, and
  the same seed produces byte-identical results on every machine. There is a test that pins
  this; a change that breaks it is a bug, not a new feature.
- New admission policies implement `AdmissionPolicy` in `policies.py` with a test in
  `tests/test_policies.py`. A policy is a pure function of the request and the observable
  gateway state — it may not read the future, and it may not read backend internals a real
  gateway could not scrape.
- New SLIs go in `metrics.py`. Percentiles use the nearest-rank method throughout; do not
  mix in interpolating implementations.
- **Fidelity claims must be honest.** This is a queueing model, not vLLM. If you add a
  behaviour, say plainly in the README what it approximates and where the approximation
  breaks. Overstating fidelity is the one thing that would make this repo worthless.
