# MIST

**MIST** is a discrete-event simulator for co-designing LLM inference
systems: heterogeneous hardware, multi-stage request pipelines
(KV-cache retrieval → prefill → decode), batching policies and
prefill/decode disaggregation. It is the artifact for the SC 2026 paper
*"MIST: A Co-Design Framework for Heterogeneous, Multi-Stage LLM Inference."*

MIST answers questions like *"what TTFT/TPOT/throughput would this model get on
2×H100 prefill + 4×L40S decode under this trace?"* in seconds, without the
hardware. Per-step latencies come from either the analytical
[GenZ](https://github.com/abhibambhaniya/GenZ-LLM-Analyzer) model or from
regressors trained on profiled vLLM runs.

> **Status: research code.** The core simulation loop is tested and used for
> the paper, but APIs are not stable, some modules are experimental (see
> [Known limitations](#known-limitations)), and results from older, pre-release
> versions can differ — see [CHANGELOG.md](CHANGELOG.md).

## Install

Requires Python ≥ 3.10 and [Git LFS](https://git-lfs.com) (the profiled vLLM
runtime tables, ~220 MB, are stored in LFS).

```sh
git lfs install
git clone https://github.com/MIST-Simulator/MIST.git
cd MIST
pip install -e ".[test]"
pytest            # ~10 s
```

This installs GenZ from its GitHub repository; the PyPI `genz_llm` release is
too old for MIST.

If you only use the analytical `PlatformConfig`, you can skip the large tables
with `GIT_LFS_SKIP_SMUDGE=1 git clone ...` and later fetch what you need with
`git lfs pull --include "mist/Platforms/vllm_runtime_data/<file>.csv"`.

## Quickstart

```python
from mist import BatchingMethod, LengthVariables, PlatformConfig, PoissonDistribution, SchedulerConfig
from mist.Coordinator import MISTCoordinator
from mist.Engine import EngineType, LLMEngine

model = "meta-llama/meta-llama-3.1-8b"

# 4 req/s for 10 s; all times in MIST are milliseconds.
requests = PoissonDistribution(
    rps=4, sim_time=10_000, rand_seed=0,
    input_vars=LengthVariables(1024, 256),   # mean, std-dev
    output_vars=LengthVariables(128, 32),
).request_queue

coordinator = MISTCoordinator(requests, logging_file=None)
types = [EngineType.PREFILL, EngineType.DECODE]
coordinator.add_engine(
    LLMEngine(
        model=model,
        engine_types=types,
        scheduler_config=SchedulerConfig(batching_method=BatchingMethod.CHUNKED, chunk_size=512),
        platform=PlatformConfig(device="H100_GPU", tensor_parallel_size=1, model=model, bits="bf16"),
    ),
    types,
)
coordinator.run_sim()

stats = coordinator.get_global_stats()
print(f"TTFT {stats.TTFT:.1f} ms, TPOT {stats.TPOT:.1f} ms, {stats.output_throughput:.0f} output tok/s")
```

Pass `logging_file="trace.json"` to write a Chrome trace you can open in
`chrome://tracing` or [Perfetto](https://ui.perfetto.dev).

More in [`examples/`](examples):

| Script | Shows |
| --- | --- |
| `aggregated_serving.py` | One replica doing prefill + decode; chunked vs continuous batching |
| `disaggregated_prefill_decode.py` | Splitwise-style prefill/decode pools with KV transfer, replaying a trace |
| `kv_cache_retrieval.py` | A cache-retrieval stage loading prefix KV from a DRAM/SSD hierarchy |

## Architecture

MIST is organised in four layers:

```
 Coordinator   event loop (heap of REQUEST_ARRIVAL / ENGINE_RUN_STEP events),
     │         request routing, inter-engine KV transfer cost
     ▼
 Engine        one model instance: LLMEngine (prefill/decode),
     │         KVRetrievalEngine, RAG engine, host
     ▼
 Scheduler     per-engine batching, adapted from vLLM: token / batch-size /
     │         KV budgets; STATIC, CONTINUOUS, MIXED, CHUNKED, DISAGGREGATED
     ▼
 Platform      cost model: how long a batch takes
               PlatformConfig       analytical (GenZ)
               vLLMPlatformConfig   RandomForest over profiled vLLM step times
               TracePlatformConfig  interpolated performance traces
               MemoryCacheConfig    tiered KV-cache retrieval
```

| Package | Contents |
| --- | --- |
| `mist.Coordinator` | `MISTCoordinator`, `MISTCoordinatorDisagg` (Splitwise), `MISTCoordinatorDisagg_DistServe`, routers |
| `mist.Engine` | `MISTEngine` base, `LLMEngine`, `KVRetrievalEngine`, `EngineMetrics` |
| `mist.Scheduler` | `Scheduler`, `SchedulerConfig`, `BatchingMethod` |
| `mist.Platforms` | cost models above; bundled vLLM profiles in `Platforms/vllm_runtime_data/` |
| `mist.Request`, `mist.Input_requests` | request lifecycle; Uniform/Poisson/Normal/Bursty generators, trace replay |
| `mist.Global_Network` | GPU interconnect specs for transfer cost |
| `mist.Tracing` | Chrome trace JSON writer |

### Things worth knowing

- **Units:** all times are milliseconds, including replayed trace arrivals.
- **Precision:** `PlatformConfig(bits=...)` defaults to `"bf16"`. Set it to what
  the deployment actually runs (`"fp8"`, `"int8"`, ...).
- **Latency cache:** `PlatformConfig` keeps computed step latencies in
  `~/.cache/mist` (override with `MIST_CACHE_DIR`) and reuses the nearest entry
  within a tolerance. Delete the directory for a cold, bit-reproducible run.
- **Reproducibility:** every request generator takes `rand_seed` and uses its
  own RNG, so parallel sweeps are safe.
- **Logging:** diagnostics go through the standard `logging` module under the
  `mist` logger, e.g. `logging.getLogger("mist").setLevel(logging.DEBUG)`.

## Reproducing the paper

Paper figures and tables are produced by
[MIST-Simulator/SC_Paper_Charts](https://github.com/MIST-Simulator/SC_Paper_Charts),
which installs this package.

## Known limitations

- **GenZ version skew.** Current GenZ `db.py` refers to `*QuantMode.bfloat16`
  names that its vendored `aiconfigurator` renamed; constructing a GenZ
  `System` for the profiled-ops devices (`h100_sxm`, `tpu_v6e`, ...) raises
  `AttributeError` until that is fixed upstream. The table-driven devices
  (`H100_GPU`, `A100_40GB_GPU`, `L40S`, ...) are unaffected.
- **Latency cache history.** Because the cache matches within a tolerance,
  results can differ in the last few significant digits between a cold and a
  warm cache.
- **Over-long prompts with non-chunked batching.** A prompt longer than
  `max_num_batched_tokens` is never scheduled and blocks the head of the
  waiting queue. Use `CHUNKED` batching or raise the budget.
- The scheduler has no preemption. Requests whose KV cache can never fit on
  their engine are dropped (`FINISHED_IGNORED`) and are not counted as served.
- `vLLMPlatformConfig` falls back to the analytical model outside the profiled
  (model, GPU, TP) combinations and warns when it does.
- The RAG engine and the DistServe coordinator see far less use than the rest
  of the code; treat their results as experimental.

## Citation

```bibtex
@inproceedings{mist_sc26,
  title     = {{MIST}: A Co-Design Framework for Heterogeneous,
               Multi-Stage {LLM} Inference},
  booktitle = {Proceedings of the International Conference for High
               Performance Computing, Networking, Storage and Analysis
               (SC)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). Contributions welcome, see
[CONTRIBUTING.md](CONTRIBUTING.md).
