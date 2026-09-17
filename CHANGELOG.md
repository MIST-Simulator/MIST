# Changelog

## 0.1.0 — first public release

Initial open-source release of the simulator previously developed as `GenA`.

### ⚠️ Changes that alter results

Numbers produced with pre-release `GenA` checkouts will not match this release
in the cases below.

- **Default precision is now bf16.** `PlatformConfig` used to pass
  `bits='fp8'` to GenZ regardless of the device. That made bf16/fp16
  deployments look about 2× faster at decode than they are, and gave them
  about 2× the KV-cache capacity. `PlatformConfig`, `vLLMPlatformConfig` and
  `TracePlatformConfig` now take `bits=`, defaulting to `'bf16'`. **To
  reproduce old numbers, pass `bits='fp8'`.** The KV-token budget of
  `LLMEngine` is derived from the platform, so it changes too, including for
  `vLLMPlatformConfig`.
- **Engines no longer share latency caches.** Every `LLMEngine` in a process
  used to share one runtime cache through a mutable default argument, so
  engines on different hardware or models returned identical step times.
  Multi-engine or multi-configuration runs in a single process are affected.
- **`TraceDistributions` arrival times are milliseconds.** They used to be
  seconds, which replayed traces 1000× too densely. (`TraceIngestion` was
  already correct.)
- **`total_token_throughput` excludes reused KV.** The old code subtracted
  `past_context` from the prompt-token count, which made the value negative for
  KV-retrieval workloads. `output_throughput` is unchanged.
- **The scheduler enforces the KV-cache budget.** It used to check the budget
  only before admitting each request, so batches overshot GPU memory, and a
  request larger than the engine's whole KV capacity was still scheduled
  (crashing GenZ with "params would not fit on chip"). Requests are now
  admitted only if the batch still fits, and requests that can never fit are
  dropped as `FINISHED_IGNORED` with a warning. Runs at high load or with long
  contexts can batch less aggressively than before.
- **The latency cache moved** from the package directory to `~/.cache/mist`
  (`MIST_CACHE_DIR`), and its filenames now include the precision. Old caches
  are not reused.

Seeded Poisson and Normal request queues are **bit-identical** to earlier
versions. `UniformDistribution` and `BurstyDistribution` were previously
unseeded and are now reproducible.

### Renamed

- Package `GenA` → `mist`, distribution `GenA_llm` → `mist`.
- `GenACoordinator` → `MISTCoordinator`, `GenACoordinatorDisagg` →
  `MISTCoordinatorDisagg`, `GenACoordinatorDisagg_DistServe` →
  `MISTCoordinatorDisagg_DistServe`, `GenAEngine` → `MISTEngine`.
- Coordinator attribute `GenA_engines` → `engines`.
- Modules `GenA_Coordinator.py` / `GenA_Engine.py` → `MIST_Coordinator.py` /
  `MIST_Engine.py`.

No compatibility aliases are provided.

### Fixed

- Mutable default arguments shared state across instances: `LLMEngine`
  caches, `engine_types`, `Request.stages`, the coordinator's request queue,
  and the default request queue and platform of the Disagg coordinators (which
  were also built at import time).
- Request generators use per-instance RNGs instead of seeding the global
  `random` / `np.random`, so queues built concurrently are reproducible.
- `print` on the simulation hot path replaced by `logging` under the `mist`
  logger.
- `TraceDistributions(...)` with extra keyword arguments raised `TypeError`.
- Latency cache CSVs are read with round-trip float precision.

### Added

- pytest suite, runnable examples, GitHub Actions CI (Python 3.10–3.12).
