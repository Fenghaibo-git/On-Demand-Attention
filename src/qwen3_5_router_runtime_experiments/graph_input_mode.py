"""Explicit graph input policy; importing this module has no device work."""
import os


def bound_graph_inputs_enabled():
    mode = os.environ.get('QWEN_GRAPH_INPUTS', 'copy')
    if mode not in ('copy', 'bound'):
        raise ValueError('QWEN_GRAPH_INPUTS must be copy or bound')
    return mode == 'bound'
