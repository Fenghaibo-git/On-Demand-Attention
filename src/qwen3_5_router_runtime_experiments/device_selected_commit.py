"""Selected KV writes with a replay-varying device position.

The original kernel remains unchanged. This JIT wrapper loads the current
position inside the graph, then calls that same bit-preserving store routine.
It is not installed by any existing runtime or GPU queue.
"""
import triton
import triton.language as tl

from .graphed_kv_commit_kernels import selected_commit_all


@triton.jit
def selected_commit_all_device_position(
        TABLE, SLOTS, POSITION,
        WRITE_FULL: tl.constexpr, BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr, ELEMENTS: tl.constexpr,
        CAPACITY: tl.constexpr, SINKS: tl.constexpr, RECENT: tl.constexpr,
        BLOCK: tl.constexpr):
    position = tl.load(POSITION).to(tl.int64)
    selected_commit_all(TABLE, SLOTS, position, WRITE_FULL, BLOCK_SIZE,
                        HEAD_DIM, ELEMENTS, CAPACITY, SINKS, RECENT, BLOCK)
