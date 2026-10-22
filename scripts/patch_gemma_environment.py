#!/usr/bin/env python3
"""Apply the pinned Transformers 5.10 optional HF DCP writer import guard.

VeOmni's FileSystemWriter checkpoint path is unaffected. This edits only the
active Python environment and refuses to patch an unknown source version.
"""
from importlib.metadata import distribution


def main():
    package = distribution("transformers")
    if not package.version.startswith("5.10.0"):
        raise RuntimeError("Use the Transformers revision in requirements/gemma.txt")
    path = package.locate_file("transformers/distributed/utils.py")
    source = path.read_text()
    old_import = '''    if is_torch_greater_or_equal("2.7"):
        from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageWriter'''
    new_import = '''    # On Demand Attention: this Torch build lacks the optional HF-format writer.
    HuggingFaceStorageWriter = None
    if is_torch_greater_or_equal("2.7"):
        try:
            from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageWriter
        except ImportError:
            pass'''
    old_guard = '''    if not is_torch_greater_or_equal("2.7"):
        raise OSError("Distributed checkpoint saving requires `torch>=2.7`.")'''
    new_guard = '''    if not is_torch_greater_or_equal("2.7") or HuggingFaceStorageWriter is None:
        raise OSError("HF-format distributed export requires torch.distributed.checkpoint.hf_storage.")'''
    if new_import in source and new_guard in source:
        print("Optional HF DCP guard already applied")
        return
    if source.count(old_import) != 1 or source.count(old_guard) != 1:
        raise RuntimeError("Unknown source content; no files changed")
    path.write_text(source.replace(old_import, new_import).replace(old_guard, new_guard))
    print("Applied optional HF DCP guard; VeOmni DCP remains enabled")


if __name__ == "__main__":
    main()
