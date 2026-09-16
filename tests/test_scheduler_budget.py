from mist import BatchingMethod, SchedulerConfig
from mist.Request import Request
from mist.Scheduler.scheduler import Scheduler
from conftest import make_queue, run_sim


def _schedule(config, input_lens):
    sched = Scheduler(config)
    reqs = [Request(request_id=i, input_len=n, output_len=4) for i, n in enumerate(input_lens)]
    for r in reqs:
        sched.add_request(r)
    prefills, decodes = sched.schedule(current_time=0)
    return reqs, prefills, decodes


def test_chunked_prefill_fills_exactly_one_chunk():
    reqs, prefills, _ = _schedule(SchedulerConfig(batching_method=BatchingMethod.CHUNKED, chunk_size=512), [400, 400, 400])
    scheduled = [r.input_len - r.remaining_prefill_tokens for r in reqs]
    assert scheduled == [400, 112, 0]
    assert prefills == reqs[:2]


def test_token_budget_without_chunking_does_not_split_prompts():
    config = SchedulerConfig(batching_method=BatchingMethod.CONTINUOUS, max_num_batched_tokens=1000)
    reqs, prefills, _ = _schedule(config, [600, 600])
    assert prefills == reqs[:1]
    assert reqs[1].remaining_prefill_tokens == 600


def test_batch_size_budget():
    config = SchedulerConfig(batching_method=BatchingMethod.CONTINUOUS, max_batch_size=2)
    _, prefills, _ = _schedule(config, [10] * 5)
    assert len(prefills) == 2


def test_budgets_hold_for_every_step_of_a_simulation():
    coord = run_sim(make_queue(rps=20), max_batch_size=4)
    engine = coord.engines[0]
    states = engine.scheduler.state_tracker
    assert states, "scheduler never ran"
    assert all(s.prefills_sched + s.decodes_sched <= 4 for s in states)
    assert coord.request_serviced == coord.request_accepted
