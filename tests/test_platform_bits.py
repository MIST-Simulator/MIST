from mist import PlatformConfig
from mist.Request import Request, RequestStage
from conftest import MODEL


def _decode_step_ms(bits):
    platform = PlatformConfig(device="H100_GPU", model=MODEL, bits=bits)
    req = Request(input_len=1024, output_len=1, request_id=0, stages=[RequestStage.DECODE])
    return platform.get_chunked_time([], [req])[0]


def test_default_precision_is_bf16():
    assert PlatformConfig(device="H100_GPU", model=MODEL).bits == "bf16"


def test_bits_reaches_cost_model():
    # Wider weights -> more memory traffic per decode step -> slower.
    assert _decode_step_ms("bf16") > _decode_step_ms("fp8")
