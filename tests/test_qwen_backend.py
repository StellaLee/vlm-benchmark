"""Qwen backend response parsing (pure, no network).

The `direct` strategy turns a sequence logprob into a confidence signal
(exp(avg_logprob)), so the backend must surface the mean token logprob from the
OpenAI-compatible response. These pin that parsing without hitting the API."""

from avbench.inference.backends.qwen import QwenClient


def _client():
    # Bypass __init__ (which needs httpx + an API key); _to_result only needs .model.
    c = object.__new__(QwenClient)
    c.model = "qwen-vl-max"
    return c


def test_to_result_averages_token_logprobs():
    data = {
        "choices": [{
            "message": {"content": "Yes"},
            "logprobs": {"content": [
                {"token": "Yes", "logprob": -0.2},
                {"token": ".", "logprob": -0.4},
            ]},
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }
    res = _client()._to_result(data)
    assert res.text == "Yes"
    assert res.avg_logprob is not None
    assert abs(res.avg_logprob - (-0.3)) < 1e-9


def test_to_result_logprobs_absent_stays_none():
    data = {"choices": [{"message": {"content": "No"}}], "usage": {}}
    res = _client()._to_result(data)
    assert res.text == "No"
    assert res.avg_logprob is None
