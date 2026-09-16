# Contributing to MIST

Bug reports, fixes and new platform/engine models are welcome.

## Setup

```sh
git lfs install
git clone https://github.com/MIST-Simulator/MIST.git && cd MIST
pip install -e ".[test]"
pytest
```

## Pull requests

- Keep each PR focused on one change, and include a test that fails without it.
  Tests live in `tests/`, and shared helpers (`make_queue`, `run_sim`) are in
  `tests/conftest.py`.
- `pytest` and the scripts in `examples/` must pass.
- If a change alters simulated numbers (latencies, throughput, KV capacity),
  say so in the PR and add an entry under "Changes that alter results" in
  `CHANGELOG.md`. People compare against published figures.
- Times are milliseconds everywhere. Anything stochastic takes a seed and uses
  its own generator, never global `random` / `np.random` state.
- Use `logging.getLogger(__name__)` for diagnostics, not `print`.
- Never use mutable objects (lists, dicts, configs) as default argument values.

## Adding data

Profiled runtime tables (`mist/Platforms/vllm_runtime_data/*.csv`) are stored
in Git LFS, and `.gitattributes` tracks them automatically. Name new tables
`<Model>_NVIDIA_<GPU>_<TP>.csv` so `vLLMPlatformConfig` can find them.

## Reporting bugs

Open an issue with a minimal script, the output you expected, what you got,
and your `mist` / GenZ versions.
