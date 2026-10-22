"""Streaming timestamps and strict decode-only timing, without GPU imports."""

from __future__ import annotations

import time


def summarize_times(token_times_ns):
    if any(b <= a for a, b in zip(token_times_ns, token_times_ns[1:])):
        raise ValueError("Token availability timestamps must strictly increase")
    count = max(0, len(token_times_ns) - 1)
    seconds = ((token_times_ns[-1] - token_times_ns[0]) / 1e9
               if count else None)
    return {"eligible": count > 0, "generated_tokens": len(token_times_ns),
            "excluded_first_tokens": min(1, len(token_times_ns)),
            "decode_tokens": count, "decode_seconds": seconds,
            "decode_tps": count / seconds if seconds else None}


def timed_request(engine, request_id, prompt, parameters, clock=time.perf_counter_ns):
    if engine.has_unfinished_requests():
        raise RuntimeError("Timing requires an empty serial engine")
    started = clock()
    engine.add_request(request_id, {"prompt_token_ids": prompt}, parameters)
    tokens, timestamps = [], []
    finished = False
    finish_reason = None
    while engine.has_unfinished_requests():
        outputs = engine.step()
        available = clock()
        if len(outputs) > 1:
            raise RuntimeError("Multiple requests in a serial timing step")
        for output in outputs:
            if len(output.outputs) != 1:
                raise RuntimeError("Require exactly one continuation")
            continuation = output.outputs[0]
            new_tokens = list(continuation.token_ids)
            if len(new_tokens) > 1:
                raise RuntimeError("Need one DELTA token per step; cannot infer missing timestamps")
            if new_tokens:
                tokens.extend(new_tokens)
                timestamps.append(available)
            if output.finished:
                finished = True
                finish_reason = continuation.finish_reason
    if not finished:
        raise RuntimeError("Request exited without final output")
    summary = summarize_times(timestamps)
    summary.update({"token_ids": tokens,
                    "token_available_ns": [t - started for t in timestamps],
                    "ttft_seconds": (timestamps[0] - started) / 1e9 if timestamps else None,
                    "wall_seconds": (clock() - started) / 1e9,
                    "finish_reason": finish_reason})
    return summary
