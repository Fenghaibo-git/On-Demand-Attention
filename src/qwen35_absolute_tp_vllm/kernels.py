"""Bounded-ring attention; optional original StreamingLLM cache-relative RoPE.

Keys are read in chronological cache order. Raw-key mode applies partial
NeoX RoPE inside the attention kernel, without gathering/rotating a whole KV
window into a temporary tensor. Absolute mode consumes already rotated keys.
"""
import triton
import triton.language as tl


@triton.jit
def partials(Q, K, V, CK, CV, COS, SIN, POSITION, P, M, L,
             H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
             SINK: tl.constexpr, RECENT: tl.constexpr, SPLITS: tl.constexpr,
             SCALE: tl.constexpr, SHIFT: tl.constexpr, ROT: tl.constexpr,
             TILE: tl.constexpr, BD: tl.constexpr):
    h, split = tl.program_id(0), tl.program_id(1)
    hk = h // (H // HK)
    length = tl.load(POSITION).to(tl.int32)
    t = split * TILE + tl.arange(0, TILE)
    d = tl.arange(0, BD)
    sinks = tl.minimum(length, SINK)
    recent = tl.minimum(tl.maximum(length - SINK, 0), RECENT)
    past = sinks + recent
    absolute = tl.maximum(SINK, length - RECENT) + t - sinks
    slot = tl.where(t < sinks, t, SINK + absolute % RECENT)
    off = (hk * (SINK + RECENT) + slot[:, None]) * D + d[None, :]
    valid = (t[:, None] < past) & (d[None, :] < D)
    k = tl.load(K + off, valid, other=0).to(tl.float32)
    ck = tl.load(CK + hk * D + d, d < D, other=0).to(tl.float32)
    k = tl.where(t[:, None] == past, ck[None, :], k)
    q = tl.load(Q + h * D + d, d < D, other=0).to(tl.float32)
    if SHIFT:
        pair = tl.where(d < ROT // 2, d + ROT // 2, d - ROT // 2)
        pair = tl.where(d < ROT, pair, d)
        poff = (hk * (SINK + RECENT) + slot[:, None]) * D + pair[None, :]
        kp = tl.load(K + poff, valid, other=0).to(tl.float32)
        ckp = tl.load(CK + hk * D + pair, d < D, other=0).to(tl.float32)
        kp = tl.where(t[:, None] == past, ckp[None, :], kp)
        qp = tl.load(Q + h * D + pair, d < D, other=0).to(tl.float32)
        freq = d % (ROT // 2)
        cosk = tl.load(COS + t[:, None] * (ROT // 2) + freq[None, :],
                       (t[:, None] <= past) & (d[None, :] < ROT), other=1)
        sink = tl.load(SIN + t[:, None] * (ROT // 2) + freq[None, :],
                       (t[:, None] <= past) & (d[None, :] < ROT), other=0)
        cosq = tl.load(COS + past * (ROT // 2) + freq, d < ROT, other=1)
        sinq = tl.load(SIN + past * (ROT // 2) + freq, d < ROT, other=0)
        sign = tl.where(d < ROT // 2, -1.0, 1.0)
        # Explicit BF16/FP16 rounding matches the independent rotation oracle.
        k = (k * cosk + sign[None, :] * kp * sink).to(CK.dtype.element_ty).to(tl.float32)
        q = (q * cosq + sign * qp * sinq).to(Q.dtype.element_ty).to(tl.float32)
    logits = tl.sum(k * q[None, :], 1) * SCALE
    logits = tl.where(t <= past, logits, -float("inf"))
    maximum = tl.max(logits, 0)
    safe = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp(logits - safe)
    normalizer = tl.sum(weights, 0)
    v = tl.load(V + off, valid, other=0).to(tl.float32)
    cv = tl.load(CV + hk * D + d, d < D, other=0).to(tl.float32)
    v = tl.where(t[:, None] == past, cv[None, :], v)
    numerator = tl.sum(weights[:, None] * v, 0)
    tl.store(P + (h * SPLITS + split) * D + d, numerator, d < D)
    tl.store(M + h * SPLITS + split, maximum)
    tl.store(L + h * SPLITS + split, normalizer)


@triton.jit
def commit(K, V, CK, CV, POSITION, D: tl.constexpr, HK: tl.constexpr,
           SINK: tl.constexpr, RECENT: tl.constexpr, B: tl.constexpr):
    x = tl.program_id(0) * B + tl.arange(0, B)
    position = tl.load(POSITION).to(tl.int32)
    slot = tl.where(position < SINK, position, SINK + position % RECENT)
    off = (x // D * (SINK + RECENT) + slot) * D + x % D
    k = tl.load(CK + x, x < HK * D, other=0)
    v = tl.load(CV + x, x < HK * D, other=0)
    tl.store(K + off, k, x < HK * D)
    tl.store(V + off, v, x < HK * D)
