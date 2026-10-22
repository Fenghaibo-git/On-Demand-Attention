#!/usr/bin/env python3
# Modified for the portable Router v4.2 release; see NOTICE.
"""Retokenize Step-3.5 raw dialogs for the Qwen3.5 hybrid Router.

The upstream Qwen3 compiled shards contain Qwen3 token IDs and therefore must
not be reused with Qwen3.5.  This builder starts from the canonical raw JSON,
implements the reference StepTronOSS last-user supervision semantics, performs
content deduplication before splitting, and creates deterministic disjoint
train/dev/test populations in the original 4K--16K length buckets.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


FORMAT_VERSION = 2
DEFAULT_BOUNDARIES = (4096, 6144, 8192, 10240, 16385)
DEFAULT_RAW_REVISION = "c994154a801557540c56af623f31b58c4770c652"
ROUTER_SINK_SIZE = 4
ROUTER_WINDOW_SIZE = 2048

_TOKENIZER = None


def candidate_contract_sha256() -> str:
    contract = {
        "version": 1,
        "source": "canonical_raw_json",
        "tokenizer": "qwen3.5_official_chat_template",
        "language_only": True,
        "last_user_only_supervision": True,
        "legacy_role_normalization": "assistant -> system+ -> assistant maps system to tool",
        "no_truncation": True,
        "content_hash": "sha256(canonical_full_raw_record_json)",
    }
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class Candidate:
    source_shard: str
    source_record_index: int
    content_sha256: str
    num_tokens: int
    loss_tokens: int
    legacy_system_as_tool_turns: int


@dataclass(frozen=True)
class Assignment:
    split: str
    bucket: int
    priority_hex: str
    content_sha256: str
    expected_num_tokens: int
    expected_loss_tokens: int
    legacy_system_as_tool_turns: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/path/to/datasets/Step-3.5-Flash-SFT/json/general"),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("/path/to/models/Qwen3.5-2B"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/path/to/datasets/processed/"
            "Step-3.5-Flash-SFT-qwen3.5-router-4k16k-"
            "content-dedup-train196608-dev8192-test8192"
        ),
    )
    parser.add_argument("--boundaries", default=",".join(map(str, DEFAULT_BOUNDARIES)))
    parser.add_argument("--train-per-bucket", type=int, default=49_152)
    parser.add_argument("--dev-per-bucket", type=int, default=2_048)
    parser.add_argument("--test-per-bucket", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=20_260_901)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--row-group-size", type=int, default=64)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument(
        "--max-source-shards",
        type=int,
        default=None,
        help="Diagnostic-only prefix of raw shards; omit for the formal build.",
    )
    parser.add_argument(
        "--max-records-per-shard",
        type=int,
        default=None,
        help="Diagnostic-only record cap per raw shard; omit for the formal build.",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Run the complete token-length/content census without writing split data.",
    )
    parser.add_argument(
        "--candidate-cache-root",
        type=Path,
        help="Optional canonical cache for the deduplicated token-length census.",
    )
    parser.add_argument(
        "--reuse-candidate-cache",
        action="store_true",
        help="Load and verify --candidate-cache-root instead of retokenizing the corpus.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def parse_boundaries(value: str) -> tuple[int, ...]:
    try:
        boundaries = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--boundaries must contain comma-separated integers.") from error
    if len(boundaries) < 2 or any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("--boundaries must be strictly increasing.")
    return boundaries


def bucket_index(length: int, boundaries: tuple[int, ...]) -> Optional[int]:
    if length < boundaries[0] or length >= boundaries[-1]:
        return None
    return int(np.searchsorted(boundaries[1:-1], length, side="right"))


def iter_json_array(path: Path) -> Iterator[tuple[int, dict]]:
    """Stream a top-level JSON array without materializing a 600-MiB shard."""

    decoder = json.JSONDecoder()
    buffer = ""
    cursor = 0
    started = False
    record_index = 0
    with path.open("r", encoding="utf-8") as handle:
        eof = False
        while True:
            if cursor:
                buffer = buffer[cursor:]
                cursor = 0
            chunk = handle.read(4 * 1024 * 1024)
            eof = not chunk
            buffer += chunk
            if not started:
                whitespace = len(buffer) - len(buffer.lstrip())
                if whitespace >= len(buffer) and not eof:
                    continue
                if whitespace >= len(buffer) or buffer[whitespace] != "[":
                    raise ValueError(f"Raw shard is not a top-level JSON array: {path}")
                cursor = whitespace + 1
                started = True

            while True:
                while cursor < len(buffer) and (buffer[cursor].isspace() or buffer[cursor] == ","):
                    cursor += 1
                if cursor < len(buffer) and buffer[cursor] == "]":
                    return
                if cursor >= len(buffer):
                    break
                try:
                    value, end = decoder.raw_decode(buffer, cursor)
                except json.JSONDecodeError:
                    break
                if not isinstance(value, dict):
                    raise TypeError(f"Raw record {path}:{record_index} is not an object.")
                yield record_index, value
                record_index += 1
                cursor = end

            if eof:
                tail = buffer[cursor:].strip()
                if tail not in {"", "]"}:
                    raise ValueError(f"Truncated JSON record at the end of {path}.")
                return


def init_worker(tokenizer_path: str) -> None:
    global _TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)


def normalize_token_ids(value) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(token, int) for token in value):
        raise TypeError(f"Unexpected tokenized output: {type(value).__name__}")
    return value


def apply_template(messages: list[dict], tools) -> list[int]:
    add_generation_prompt = messages[-1]["role"].lower() != "assistant"
    kwargs = {
        "conversation": messages,
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        kwargs["tools"] = tools
    return normalize_token_ids(_TOKENIZER.apply_chat_template(**kwargs))


def normalize_messages_for_qwen3_5(messages: list[dict]) -> list[dict]:
    """Adapt legacy Step-3.5 tool traces to Qwen3.5's chat contract.

    The raw corpus uses a post-initial ``system`` turn for shell/environment
    responses.  Qwen3.5 names the same semantic role ``tool``.  Validation
    admits this conversion only for an assistant -> system+ -> assistant
    pattern, so arbitrary system instructions are never reclassified.
    """

    normalized = copy.deepcopy(messages)
    for index, message in enumerate(normalized):
        message["role"] = str(message["role"]).lower()
        if index > 0 and message["role"] == "system":
            message["role"] = "tool"
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function", tool_call)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                parsed = json.loads(arguments)
                if not isinstance(parsed, dict):
                    raise TypeError("Qwen3.5 tool-call arguments must decode to an object.")
                function["arguments"] = parsed
    return normalized


def validate_dialog(record: dict) -> Optional[str]:
    messages = record.get("conversations")
    if not isinstance(messages, list) or len(messages) < 2:
        return "less_than_two_messages"
    if not isinstance(messages[-1], dict) or str(messages[-1].get("role", "")).lower() != "assistant":
        return "last_message_not_assistant"
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return "message_not_object"
        role = str(message.get("role", "")).lower()
        if not role:
            return "missing_role"
        if role == "system" and index != 0:
            previous_role = str(messages[index - 1].get("role", "")).lower()
            next_role = (
                str(messages[index + 1].get("role", "")).lower()
                if index + 1 < len(messages) and isinstance(messages[index + 1], dict)
                else ""
            )
            if previous_role not in {"assistant", "system"} or next_role not in {
                "assistant",
                "system",
            }:
                return "unsupported_late_system_turn"
        if "content" not in message and "value" not in message:
            return "missing_content"
        content = message.get("content", message.get("value"))
        if not isinstance(content, str):
            # This experiment is explicitly language-only.  Vision/media rows
            # are not silently converted into text placeholders.
            return "non_text_content"
        if role == "assistant" and not message.get("content") and not message.get("tool_calls"):
            return "empty_assistant"
        if "\n" in str(message.get("name", "") or ""):
            return "name_contains_newline"
    return None


def legacy_system_as_tool_turns(record: dict) -> int:
    return sum(
        index > 0 and str(message.get("role", "")).lower() == "system"
        for index, message in enumerate(record["conversations"])
    )


def tokenize_dialog(record: dict) -> tuple[np.ndarray, np.ndarray]:
    """Match the official StepTronOSS HuggingFaceTemplate loss semantics."""

    messages = normalize_messages_for_qwen3_5(record["conversations"])
    tools = messages[0].get("tool_schemas") or messages[0].get("tools")
    tokens = apply_template(messages, tools)
    last_user = max(
        (index for index, message in enumerate(messages) if str(message["role"]).lower() == "user"),
        default=-1,
    )
    if last_user < 0:
        raise ValueError("No user turn found.")

    prefix = apply_template(messages[: last_user + 1], tools)
    loss_mask = np.zeros(len(tokens), dtype=np.uint8)
    current = len(prefix)
    for index in range(last_user + 1, len(messages)):
        # The final prefix is exactly the already-tokenized full dialog.  Most
        # records have one post-user assistant turn, so this avoids a duplicate
        # full-record tokenization without changing the reference boundaries.
        through_current = (
            tokens
            if index == len(messages) - 1
            else apply_template(messages[: index + 1], tools)
        )
        end = len(through_current)
        if end < current or end > len(tokens):
            raise ValueError("Chat-template prefix is not monotonic.")
        message = messages[index]
        if str(message["role"]).lower() == "assistant" and float(message.get("loss_mask", 1)) == 1.0:
            loss_mask[current:end] = 1
        current = end
    token_array = np.asarray(tokens, dtype=np.int32)
    if token_array.ndim != 1 or loss_mask.shape != token_array.shape:
        raise ValueError("Token/loss-mask shape mismatch.")
    if len(token_array) == 0 or token_array.min() < 0 or token_array.max() >= len(_TOKENIZER):
        raise ValueError("Token IDs fall outside the Qwen3.5 tokenizer vocabulary.")
    return token_array, loss_mask


def canonical_content_sha256(record: dict) -> str:
    serialized = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def scan_shard(task: tuple[str, tuple[int, ...], Optional[int]]) -> dict:
    shard_name, boundaries, record_limit = task
    path = Path(shard_name)
    candidates = []
    invalid = Counter()
    total = 0
    tokenized = 0
    outside = 0
    normalized_records = 0
    normalized_turns = 0
    for index, record in iter_json_array(path):
        if record_limit is not None and index >= record_limit:
            break
        total += 1
        error = validate_dialog(record)
        if error is not None:
            invalid[error] += 1
            continue
        try:
            tokens, loss_mask = tokenize_dialog(record)
        except Exception as error:
            raise RuntimeError(f"Tokenization failed at {path}:{index}: {error}") from error
        tokenized += 1
        converted_turns = legacy_system_as_tool_turns(record)
        normalized_records += int(converted_turns > 0)
        normalized_turns += converted_turns
        bucket = bucket_index(len(tokens), boundaries)
        if bucket is None:
            outside += 1
            continue
        candidates.append(
            Candidate(
                source_shard=path.name,
                source_record_index=index,
                content_sha256=canonical_content_sha256(record),
                num_tokens=len(tokens),
                loss_tokens=int(loss_mask.sum()),
                legacy_system_as_tool_turns=converted_turns,
            )
        )
    return {
        "source_shard": path.name,
        "total_records": total,
        "valid_tokenized_records": tokenized,
        "outside_length_range": outside,
        "legacy_system_as_tool_records": normalized_records,
        "legacy_system_as_tool_turns": normalized_turns,
        "invalid": dict(invalid),
        "candidates": candidates,
    }


def selection_priority(seed: int, content_sha256: str) -> bytes:
    return hashlib.sha256(
        f"qwen3.5-router-split-v1\0{seed}\0{content_sha256}".encode("ascii")
    ).digest()


def assign_splits(
    candidates_by_bucket: list[list[Candidate]],
    seed: int,
    train_per_bucket: int,
    dev_per_bucket: int,
    test_per_bucket: int,
) -> tuple[dict[str, dict[int, Assignment]], dict, dict]:
    assignments: dict[str, dict[int, Assignment]] = defaultdict(dict)
    split_stats = {name: [0] * len(candidates_by_bucket) for name in ("train", "dev", "test")}
    split_normalization = {
        name: {"records": 0, "turns": 0} for name in ("train", "dev", "test")
    }
    held_out_per_bucket = dev_per_bucket + test_per_bucket
    train_capacity = []
    for bucket, values in enumerate(candidates_by_bucket):
        if len(values) < held_out_per_bucket:
            raise RuntimeError(
                f"Bucket {bucket} has {len(values):,} unique candidates; "
                f"{held_out_per_bucket:,} are required for frozen dev/test."
            )
        train_capacity.append(len(values) - held_out_per_bucket)

    requested_train_rows = train_per_bucket * len(candidates_by_bucket)
    if sum(train_capacity) < requested_train_rows:
        raise RuntimeError(
            f"The deduplicated population has capacity for {sum(train_capacity):,} "
            f"training rows after frozen dev/test; {requested_train_rows:,} are required."
        )

    # Preserve the fixed per-bucket dev/test populations.  A tokenizer-specific
    # shortage in one training bucket is capped at its available capacity, then
    # redistributed one row at a time in stable bucket order.  This retains the
    # requested total train size without duplicates or moving held-out samples.
    train_counts = [min(train_per_bucket, capacity) for capacity in train_capacity]
    deficit = requested_train_rows - sum(train_counts)
    while deficit:
        progressed = False
        for bucket, capacity in enumerate(train_capacity):
            if train_counts[bucket] >= capacity:
                continue
            train_counts[bucket] += 1
            deficit -= 1
            progressed = True
            if deficit == 0:
                break
        if not progressed:  # Guard the capacity proof above against regressions.
            raise RuntimeError("Unable to redistribute the requested training rows.")

    for bucket, values in enumerate(candidates_by_bucket):
        ordered = sorted(
            values,
            key=lambda value: (
                selection_priority(seed, value.content_sha256),
                value.content_sha256,
                value.source_shard,
                value.source_record_index,
            ),
        )
        required = held_out_per_bucket + train_counts[bucket]
        # Held-out populations are allocated first so their identity is stable
        # if a later experiment changes only the train population size.
        intervals = (
            ("dev", 0, dev_per_bucket),
            ("test", dev_per_bucket, dev_per_bucket + test_per_bucket),
            ("train", dev_per_bucket + test_per_bucket, required),
        )
        for split, start, end in intervals:
            for candidate in ordered[start:end]:
                priority = selection_priority(seed, candidate.content_sha256).hex()
                assignments[candidate.source_shard][candidate.source_record_index] = Assignment(
                    split=split,
                    bucket=bucket,
                    priority_hex=priority,
                    content_sha256=candidate.content_sha256,
                    expected_num_tokens=candidate.num_tokens,
                    expected_loss_tokens=candidate.loss_tokens,
                    legacy_system_as_tool_turns=candidate.legacy_system_as_tool_turns,
                )
                split_stats[split][bucket] += 1
                if candidate.legacy_system_as_tool_turns:
                    split_normalization[split]["records"] += 1
                    split_normalization[split]["turns"] += candidate.legacy_system_as_tool_turns
    return assignments, split_stats, split_normalization


def output_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("source_shard", pa.string(), nullable=False),
            pa.field("source_record_index", pa.int32(), nullable=False),
            pa.field("content_sha256", pa.string(), nullable=False),
            pa.field("selection_priority", pa.string(), nullable=False),
            pa.field("length_bucket", pa.int8(), nullable=False),
            pa.field("num_tokens", pa.int32(), nullable=False),
            pa.field("input_ids", pa.list_(pa.int32()), nullable=False),
            pa.field("loss_mask", pa.list_(pa.uint8()), nullable=False),
        ]
    )


def write_rows(path: Path, rows: list[dict], compression: str, row_group_size: int) -> None:
    schema = output_schema()
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(
        table,
        path,
        compression=compression,
        row_group_size=row_group_size,
        write_statistics=True,
    )


def upstream_manifest_path(source_root: Path) -> Path:
    path = source_root.parent.parent / ".step35-download" / (
        f"manifest-{DEFAULT_RAW_REVISION}.json"
    )
    if not path.is_file():
        path = Path("/path/to/datasets/Step-3.5-Flash-SFT/.step35-download") / (
            f"manifest-{DEFAULT_RAW_REVISION}.json"
        )
    if not path.is_file():
        raise FileNotFoundError("Pinned Step-3.5 upstream manifest is missing.")
    return path


def materialize_shard(
    task: tuple[str, dict[int, Assignment], str, str, int, str, Optional[int]]
) -> list[dict]:
    (
        shard_path_value,
        selected,
        staging_root_value,
        compression,
        row_group_size,
        tokenizer_path,
        record_limit,
    ) = task
    del tokenizer_path  # tokenizer is initialized once in this worker process.
    shard_path = Path(shard_path_value)
    staging_root = Path(staging_root_value)
    rows_by_split: dict[str, list[dict]] = defaultdict(list)
    found = set()
    for index, record in iter_json_array(shard_path):
        if record_limit is not None and index >= record_limit:
            break
        assignment = selected.get(index)
        if assignment is None:
            continue
        tokens, loss_mask = tokenize_dialog(record)
        observed_digest = canonical_content_sha256(record)
        if observed_digest != assignment.content_sha256:
            raise RuntimeError(f"Content changed between census/materialization at {shard_path}:{index}.")
        if len(tokens) != assignment.expected_num_tokens or int(loss_mask.sum()) != assignment.expected_loss_tokens:
            raise RuntimeError(f"Tokenization changed between passes at {shard_path}:{index}.")
        rows_by_split[assignment.split].append(
            {
                "source_shard": shard_path.name,
                "source_record_index": index,
                "content_sha256": observed_digest,
                "selection_priority": assignment.priority_hex,
                "length_bucket": assignment.bucket,
                "num_tokens": len(tokens),
                "input_ids": tokens,
                "loss_mask": loss_mask,
            }
        )
        found.add(index)
    missing = set(selected).difference(found)
    if missing:
        raise RuntimeError(f"Selected records disappeared from {shard_path}: {sorted(missing)[:10]}")

    outputs = []
    for split, rows in sorted(rows_by_split.items()):
        output_path = staging_root / split / "data" / f"{shard_path.stem}.parquet"
        write_rows(output_path, rows, compression, row_group_size)
        outputs.append(
            {
                "split": split,
                "path": str(output_path.relative_to(staging_root)),
                "rows": len(rows),
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            }
        )
    return outputs


def tokenizer_manifest(tokenizer_path: Path) -> dict:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "config.json",
    )
    files = {}
    for name in names:
        path = tokenizer_path / name
        if path.is_file():
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "path": str(tokenizer_path.resolve()),
        "files": files,
        "fingerprint_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def write_candidate_cache(
    cache_root: Path,
    candidates_by_bucket: list[list[Candidate]],
    census: dict,
    source_root: Path,
    tokenizer_path: Path,
    raw_shards: list[Path],
) -> None:
    if cache_root.exists():
        raise FileExistsError(f"Refusing to overwrite candidate cache: {cache_root}")
    staging = cache_root.parent / f".{cache_root.name}.incomplete"
    if staging.exists():
        raise FileExistsError(f"Incomplete candidate cache already exists: {staging}")
    staging.mkdir(parents=True)
    rows = []
    for bucket, values in enumerate(candidates_by_bucket):
        rows.extend(
            {
                "bucket": bucket,
                "source_shard": value.source_shard,
                "source_record_index": value.source_record_index,
                "content_sha256": value.content_sha256,
                "num_tokens": value.num_tokens,
                "loss_tokens": value.loss_tokens,
                "legacy_system_as_tool_turns": value.legacy_system_as_tool_turns,
            }
            for value in values
        )
    candidate_path = staging / "candidates.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, candidate_path, compression="zstd", row_group_size=4096)
    tokenizer = tokenizer_manifest(tokenizer_path)
    manifest = {
        "format": "qwen3.5-step35-router-candidate-cache-v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_format_version": FORMAT_VERSION,
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "candidate_contract_sha256": candidate_contract_sha256(),
        "source": {
            "root": str(source_root),
            "revision": DEFAULT_RAW_REVISION,
            "upstream_manifest_sha256": sha256_file(upstream_manifest_path(source_root)),
            "raw_shards": [
                {"name": path.name, "bytes": path.stat().st_size} for path in raw_shards
            ],
        },
        "tokenizer": tokenizer,
        "boundaries_half_open": census["boundaries_half_open"],
        "census": census,
        "candidate_file": {
            "path": candidate_path.name,
            "rows": table.num_rows,
            "bytes": candidate_path.stat().st_size,
            "sha256": sha256_file(candidate_path),
        },
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    atomic_write_text(staging / "manifest.json", manifest_text)
    readme = """# Qwen3.5 Step-3.5 Router candidate census cache

CPU-only deterministic cache of the complete Qwen3.5 token-length census over
the pinned Step-3.5 raw JSON. It stores only source identities, content hashes,
token/loss lengths, length buckets, and audited legacy-role conversion counts;
it does not duplicate raw dialogs or tokenized training payloads.

The formal dataset builder verifies the source revision, source shard inventory,
tokenizer fingerprint, builder hash, Parquet hash, row counts, and bucket counts
before reuse.
"""
    atomic_write_text(staging / "README.md", readme)
    success = {
        "status": "passed",
        "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "candidate_sha256": manifest["candidate_file"]["sha256"],
        "candidate_rows": table.num_rows,
    }
    atomic_write_text(staging / "_SUCCESS", json.dumps(success, indent=2, sort_keys=True) + "\n")
    os.replace(staging, cache_root)


def load_candidate_cache(
    cache_root: Path,
    source_root: Path,
    tokenizer_path: Path,
    raw_shards: list[Path],
    boundaries: tuple[int, ...],
) -> tuple[list[list[Candidate]], dict]:
    manifest_path = cache_root / "manifest.json"
    success_path = cache_root / "_SUCCESS"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or success.get("status") != "passed":
        raise ValueError("Candidate cache does not have complete/pass markers.")
    if success.get("manifest_sha256") != hashlib.sha256(manifest_text.encode("utf-8")).hexdigest():
        raise ValueError("Candidate cache manifest hash mismatch.")
    expected = {
        "builder_format_version": FORMAT_VERSION,
        "candidate_contract_sha256": candidate_contract_sha256(),
        "source_root": str(source_root),
        "revision": DEFAULT_RAW_REVISION,
        "upstream_manifest_sha256": sha256_file(upstream_manifest_path(source_root)),
        "raw_shards": [{"name": path.name, "bytes": path.stat().st_size} for path in raw_shards],
        "tokenizer_fingerprint": tokenizer_manifest(tokenizer_path)["fingerprint_sha256"],
        "boundaries": list(boundaries),
    }
    observed = {
        "builder_format_version": manifest.get("builder_format_version"),
        "candidate_contract_sha256": manifest.get("candidate_contract_sha256"),
        "source_root": manifest.get("source", {}).get("root"),
        "revision": manifest.get("source", {}).get("revision"),
        "upstream_manifest_sha256": manifest.get("source", {}).get("upstream_manifest_sha256"),
        "raw_shards": manifest.get("source", {}).get("raw_shards"),
        "tokenizer_fingerprint": manifest.get("tokenizer", {}).get("fingerprint_sha256"),
        "boundaries": manifest.get("boundaries_half_open"),
    }
    if observed != expected:
        raise ValueError(f"Candidate cache contract mismatch: {observed!r} != {expected!r}")
    candidate_meta = manifest["candidate_file"]
    candidate_path = cache_root / candidate_meta["path"]
    if candidate_path.stat().st_size != candidate_meta["bytes"]:
        raise ValueError("Candidate cache byte count mismatch.")
    if sha256_file(candidate_path) != candidate_meta["sha256"]:
        raise ValueError("Candidate cache Parquet hash mismatch.")
    table = pq.read_table(candidate_path)
    if table.num_rows != candidate_meta["rows"] or table.num_rows != success["candidate_rows"]:
        raise ValueError("Candidate cache row count mismatch.")
    candidates_by_bucket = [[] for _ in range(len(boundaries) - 1)]
    seen_contents = set()
    for row in table.to_pylist():
        candidate = Candidate(
            source_shard=row["source_shard"],
            source_record_index=int(row["source_record_index"]),
            content_sha256=row["content_sha256"],
            num_tokens=int(row["num_tokens"]),
            loss_tokens=int(row["loss_tokens"]),
            legacy_system_as_tool_turns=int(row["legacy_system_as_tool_turns"]),
        )
        bucket = int(row["bucket"])
        if bucket_index(candidate.num_tokens, boundaries) != bucket:
            raise ValueError("Candidate cache bucket/length mismatch.")
        if candidate.content_sha256 in seen_contents:
            raise ValueError("Candidate cache contains duplicate content.")
        seen_contents.add(candidate.content_sha256)
        candidates_by_bucket[bucket].append(candidate)
    observed_buckets = [len(values) for values in candidates_by_bucket]
    if observed_buckets != manifest["census"]["unique_candidates_per_bucket"]:
        raise ValueError("Candidate cache bucket counts mismatch.")
    return candidates_by_bucket, manifest["census"]


def summarize_split(root: Path, files: list[dict]) -> dict:
    rows = input_tokens = loss_tokens = eligible_tokens = 0
    buckets = Counter()
    content = set()
    ordered = hashlib.sha256()
    for item in sorted(files, key=lambda value: value["path"]):
        table = pq.read_table(root / item["path"])
        for row in table.to_pylist():
            rows += 1
            input_tokens += int(row["num_tokens"])
            mask = row["loss_mask"]
            loss_tokens += int(sum(mask))
            eligible_tokens += int(sum(mask[ROUTER_SINK_SIZE + ROUTER_WINDOW_SIZE + 1 :]))
            buckets[int(row["length_bucket"])] += 1
            digest = row["content_sha256"]
            if digest in content:
                raise RuntimeError(f"Duplicate content inside split {item['split']}: {digest}")
            content.add(digest)
            ordered.update(f"{digest}\n".encode("ascii"))
    return {
        "rows": rows,
        "input_tokens": input_tokens,
        "loss_tokens": loss_tokens,
        "router_eligible_tokens": eligible_tokens,
        "rows_per_bucket": dict(sorted(buckets.items())),
        "unique_contents": len(content),
        "ordered_content_sha256": ordered.hexdigest(),
        "content_set": content,
    }


def main() -> None:
    args = parse_args()
    boundaries = parse_boundaries(args.boundaries)
    if args.workers < 1 or args.row_group_size < 1:
        raise ValueError("Worker and row-group counts must be positive.")
    if min(args.train_per_bucket, args.dev_per_bucket, args.test_per_bucket) < 0:
        raise ValueError("Split counts cannot be negative.")
    if args.train_per_bucket < 1:
        raise ValueError("The train population must be non-empty.")
    source_root = args.source_root.resolve()
    tokenizer_path = args.tokenizer_path.resolve()
    output_root = args.output_root.resolve()
    raw_shards = sorted(source_root.glob("chunk_*.json"), key=lambda path: int(path.stem.split("_")[-1]))
    if args.max_source_shards is not None:
        raw_shards = raw_shards[: args.max_source_shards]
    if not raw_shards:
        raise FileNotFoundError(f"No raw JSON shards found in {source_root}.")
    if not (tokenizer_path / "tokenizer.json").is_file():
        raise FileNotFoundError(f"Tokenizer assets are incomplete: {tokenizer_path}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    cache_root = args.candidate_cache_root.resolve() if args.candidate_cache_root else None
    if args.reuse_candidate_cache and cache_root is None:
        raise ValueError("--reuse-candidate-cache requires --candidate-cache-root.")
    if args.reuse_candidate_cache and (
        args.max_source_shards is not None or args.max_records_per_shard is not None
    ):
        raise ValueError("A formal candidate cache cannot be combined with diagnostic source limits.")
    if args.reuse_candidate_cache:
        candidates_by_bucket, census = load_candidate_cache(
            cache_root, source_root, tokenizer_path, raw_shards, boundaries
        )
        print(f"verified_candidate_cache={cache_root}", flush=True)
    else:
        scan_tasks = [(str(path), boundaries, args.max_records_per_shard) for path in raw_shards]
        scan_results = []
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(str(tokenizer_path),),
        ) as pool:
            for completed, result in enumerate(pool.map(scan_shard, scan_tasks), start=1):
                scan_results.append(result)
                print(
                    f"census {completed}/{len(scan_tasks)} {result['source_shard']} "
                    f"records={result['total_records']:,} candidates={len(result['candidates']):,}",
                    flush=True,
                )

        candidates_by_bucket = [[] for _ in range(len(boundaries) - 1)]
        seen_contents = set()
        duplicates = [0] * len(candidates_by_bucket)
        for result in scan_results:
            for candidate in result.pop("candidates"):
                bucket = bucket_index(candidate.num_tokens, boundaries)
                assert bucket is not None
                if candidate.content_sha256 in seen_contents:
                    duplicates[bucket] += 1
                    continue
                seen_contents.add(candidate.content_sha256)
                candidates_by_bucket[bucket].append(candidate)

        census = {
            "format_version": FORMAT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "boundaries_half_open": list(boundaries),
            "source_shards": len(raw_shards),
            "source_records": sum(value["total_records"] for value in scan_results),
            "valid_tokenized_records": sum(value["valid_tokenized_records"] for value in scan_results),
            "outside_length_range": sum(value["outside_length_range"] for value in scan_results),
            "legacy_system_as_tool_records": sum(
                value["legacy_system_as_tool_records"] for value in scan_results
            ),
            "legacy_system_as_tool_turns": sum(
                value["legacy_system_as_tool_turns"] for value in scan_results
            ),
            "invalid_records": dict(
                sum((Counter(value["invalid"]) for value in scan_results), Counter())
            ),
            "unique_candidates_per_bucket": [len(values) for values in candidates_by_bucket],
            "duplicate_candidates_per_bucket": duplicates,
            "unique_candidate_legacy_system_as_tool_records_per_bucket": [
                sum(candidate.legacy_system_as_tool_turns > 0 for candidate in values)
                for values in candidates_by_bucket
            ],
            "unique_candidate_legacy_system_as_tool_turns_per_bucket": [
                sum(candidate.legacy_system_as_tool_turns for candidate in values)
                for values in candidates_by_bucket
            ],
            "diagnostic_limits": {
                "max_source_shards": args.max_source_shards,
                "max_records_per_shard": args.max_records_per_shard,
            },
        }
        if cache_root is not None:
            write_candidate_cache(
                cache_root,
                candidates_by_bucket,
                census,
                source_root,
                tokenizer_path,
                raw_shards,
            )
            print(f"wrote_candidate_cache={cache_root}", flush=True)
    print(json.dumps(census, indent=2, sort_keys=True), flush=True)
    if args.scan_only:
        return

    assignments, expected_split_stats, selected_normalization = assign_splits(
        candidates_by_bucket,
        args.seed,
        args.train_per_bucket,
        args.dev_per_bucket,
        args.test_per_bucket,
    )
    staging_root = output_root.parent / f".{output_root.name}.incomplete"
    if staging_root.exists():
        raise FileExistsError(f"Incomplete output already exists: {staging_root}")
    for split in ("train", "dev", "test"):
        (staging_root / split / "data").mkdir(parents=True)

    tasks = [
        (
            str(path),
            assignments.get(path.name, {}),
            str(staging_root),
            args.compression,
            args.row_group_size,
            str(tokenizer_path),
            args.max_records_per_shard,
        )
        for path in raw_shards
        if assignments.get(path.name)
    ]
    output_files = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(str(tokenizer_path),),
    ) as pool:
        for completed, values in enumerate(pool.map(materialize_shard, tasks), start=1):
            output_files.extend(values)
            print(f"materialize {completed}/{len(tasks)} files={len(values)}", flush=True)

    summaries = {}
    content_sets = {}
    for split in ("train", "dev", "test"):
        split_files = [value for value in output_files if value["split"] == split]
        summary = summarize_split(staging_root, split_files)
        content_sets[split] = summary.pop("content_set")
        summaries[split] = summary
        observed = [int(summary["rows_per_bucket"].get(index, 0)) for index in range(len(boundaries) - 1)]
        if observed != expected_split_stats[split]:
            raise RuntimeError(f"{split} bucket counts mismatch: {observed} != {expected_split_stats[split]}")
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = content_sets[left].intersection(content_sets[right])
        if overlap:
            raise RuntimeError(f"{left}/{right} content overlap: {len(overlap)}")

    upstream_manifest = upstream_manifest_path(source_root)
    manifest = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Qwen3.5-2B hybrid three-stream Router train/dev/test corpus",
        "source": {
            "root": str(source_root),
            "dataset": "stepfun-ai/Step-3.5-Flash-SFT",
            "revision": DEFAULT_RAW_REVISION,
            "upstream_manifest_sha256": sha256_file(upstream_manifest),
            "raw_shards": [path.name for path in raw_shards],
        },
        "tokenizer": tokenizer_manifest(tokenizer_path),
        "template_semantics": {
            "reference": "StepTronOSS HuggingFaceTemplate",
            "last_user_only": True,
            "assistant_loss_mask_required": 1,
            "no_truncation": True,
            "full_retokenization_from_raw_json": True,
            "legacy_role_normalization": {
                "source_role": "post-initial system",
                "target_role": "tool",
                "admission_pattern": "assistant -> system+ -> assistant",
                "reason": "Step-3.5 shell/environment response compatibility with the official Qwen3.5 template",
                "selected_by_split": selected_normalization,
            },
        },
        "selection": {
            "seed": args.seed,
            "priority": "sha256(qwen3.5-router-split-v1\\0seed\\0content_sha256)",
            "content_hash": "sha256(canonical_full_raw_record_json)",
            "boundaries_half_open": list(boundaries),
            "allocation_order": ["dev", "test", "train"],
            "train_per_bucket": args.train_per_bucket,
            "requested_train_per_bucket": args.train_per_bucket,
            "resolved_train_per_bucket": expected_split_stats["train"],
            "train_shortage_policy": (
                "cap tokenizer-specific short buckets after frozen dev/test, then "
                "redistribute the total deficit one row at a time in stable bucket order"
            ),
            "dev_per_bucket": args.dev_per_bucket,
            "test_per_bucket": args.test_per_bucket,
        },
        "router_eligibility": {
            "sink_size": ROUTER_SINK_SIZE,
            "window_size_including_current": ROUTER_WINDOW_SIZE,
            "eligible_loss_mask_start_index_zero_based": ROUTER_SINK_SIZE
            + ROUTER_WINDOW_SIZE
            + 1,
        },
        "census": census,
        "candidate_cache": (
            {
                "root": str(cache_root),
                "manifest_sha256": sha256_file(cache_root / "manifest.json"),
                "reused": bool(args.reuse_candidate_cache),
            }
            if cache_root is not None
            else None
        ),
        "splits": summaries,
        "files": sorted(output_files, key=lambda value: value["path"]),
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    atomic_write_text(staging_root / "manifest.json", manifest_text)
    readme = f"""# Qwen3.5 Step-3.5 Router corpus

This entity was retokenized from the canonical raw Step-3.5-Flash-SFT JSON with
`/path/to/models/Qwen3.5-2B`; it does not reuse Qwen3 token IDs.  It follows the
official StepTronOSS last-user supervision rule and contains deterministic,
content-disjoint train/dev/test splits over half-open token buckets
`{list(boundaries)}`.  No record is truncated.

Legacy shell/environment responses encoded as post-initial `system` turns are
mapped to Qwen3.5's equivalent `tool` role only when they occur in an
`assistant -> system+ -> assistant` pattern.  The manifest records source and
selected conversion counts.

- Train: {summaries['train']['rows']:,} rows, {summaries['train']['input_tokens']:,} input tokens,
  {summaries['train']['router_eligible_tokens']:,} Router-eligible targets.
- Train rows by length bucket: {expected_split_stats['train']} (the requested
  total is preserved; tokenizer-specific bucket shortages are deterministically
  redistributed after freezing equal dev/test populations).
- Dev: {summaries['dev']['rows']:,} rows, {summaries['dev']['input_tokens']:,} input tokens.
- Test: {summaries['test']['rows']:,} rows, {summaries['test']['input_tokens']:,} input tokens.
- Content overlap across every pair of splits: zero.
- Sink/window used only for eligible-token accounting: {ROUTER_SINK_SIZE}/{ROUTER_WINDOW_SIZE}.

VeOmni should read `train/data/`.  `manifest.json` records tokenizer/source
hashes, deterministic selection, complete file hashes, counts, and audit data.
"""
    atomic_write_text(staging_root / "README.md", readme)
    success = {
        "status": "passed",
        "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "split_rows": {name: value["rows"] for name, value in summaries.items()},
        "pairwise_content_overlap": 0,
    }
    atomic_write_text(staging_root / "_SUCCESS", json.dumps(success, indent=2, sort_keys=True) + "\n")
    os.replace(staging_root, output_root)
    print(json.dumps(success, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
