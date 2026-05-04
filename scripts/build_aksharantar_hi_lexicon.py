#!/usr/bin/env python3
"""
Export Aksharantar Hindi (hin) word pairs → TSV for the runtime lexicon loader.

Downloads hin.zip from Hugging Face (MohammedABU/Aksharantar) unless --input is set.
Writes: roman TAB native [TAB score] (dedup keeps max score).

Usage:
  python scripts/build_aksharantar_hi_lexicon.py
  python scripts/build_aksharantar_hi_lexicon.py --output data/lexicons/aksharantar_hi.tsv
  python scripts/build_aksharantar_hi_lexicon.py --input /path/to/unzipped/hin_train.json

Requires: huggingface_hub
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_json_under(root: Path) -> List[Path]:
    return sorted(root.rglob("*.json"))


def ingest_records(rows: List[Dict[str, Any]]) -> Dict[str, tuple]:
    """Map normalized roman → (score, native)."""
    best: Dict[str, tuple] = {}
    roman_keys = ("english word", "english_word", "roman", "src", "source_word")
    native_keys = ("native word", "native_word", "native", "tgt", "tgt_word", "hindi")

    for row in rows:
        if not isinstance(row, dict):
            continue
        roman = ""
        native = ""
        for k in roman_keys:
            if k in row and row[k]:
                roman = str(row[k]).strip()
                break
        else:
            for k, v in row.items():
                if "english" in k.lower() or k.lower() == "roman":
                    roman = str(v).strip()
                    break
        for k in native_keys:
            if k in row and row[k]:
                native = str(row[k]).strip()
                break
        else:
            for k, v in row.items():
                if "native" in k.lower() or k.lower() in ("hindi", "devanagari"):
                    native = str(v).strip()
                    break

        if not roman or not native:
            continue
        try:
            sc = float(row.get("score", 0))
        except (TypeError, ValueError):
            sc = 0.0

        rk = roman.lower().strip()
        prev = best.get(rk)
        if prev is None or sc >= prev[0]:
            best[rk] = (sc, native)
    return best


def records_from_json_file(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    def from_parsed(raw: Any) -> List[Dict[str, Any]]:
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
        if isinstance(raw, dict):
            if "data" in raw and isinstance(raw["data"], list):
                return [x for x in raw["data"] if isinstance(x, dict)]
            return [raw]
        return []

    try:
        return from_parsed(json.loads(text))
    except json.JSONDecodeError:
        rows: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError:
                continue
        return rows


def main() -> int:
    os.environ.setdefault(
        "HF_HOME", str(project_root() / ".cache" / "huggingface")
    )

    ap = argparse.ArgumentParser(description="Build Aksharantar Hindi lexicon TSV")
    ap.add_argument(
        "--output",
        type=Path,
        default=project_root() / "data" / "lexicons" / "aksharantar_hi.tsv",
        help="Output .tsv path",
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional path to exploded hin folder or hin_train.json",
    )
    ap.add_argument(
        "--repo-id",
        default="MohammedABU/Aksharantar",
        help="HF dataset repo containing hin.zip",
    )
    ap.add_argument(
        "--filename",
        default="hin.zip",
        help="ZIP filename inside the repo",
    )
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    accumulated: Dict[str, tuple] = {}

    def merge_from_json_paths(paths: List[Path]) -> None:
        for jf in paths:
            rows = records_from_json_file(jf)
            part = ingest_records(rows)
            for k, pair in part.items():
                prev = accumulated.get(k)
                if prev is None or pair[0] >= prev[0]:
                    accumulated[k] = pair

    if args.input:
        p = Path(args.input)
        if p.is_file() and p.suffix.lower() == ".json":
            merge_from_json_paths([p])
        elif p.is_dir():
            json_files = find_json_under(p)
            if not json_files:
                print(f"No JSON under {p}", file=sys.stderr)
                return 1
            merge_from_json_paths(json_files)
        else:
            print(f"Bad --input {p}", file=sys.stderr)
            return 1
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
            return 1
        zp = hf_hub_download(repo_id=args.repo_id, filename=args.filename, repo_type="dataset")
        td = tempfile.mkdtemp(prefix="aksh_hi_")
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(td)
            json_files = find_json_under(Path(td))
            if not json_files:
                print("No JSON found after extracting hin.zip", file=sys.stderr)
                return 1
            merge_from_json_paths(json_files)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# roman_native pairs from Aksharantar Hindi; built by scripts/build_aksharantar_hi_lexicon.py\n")
        for k in sorted(accumulated):
            score, native = accumulated[k]
            f.write(f"{k}\t{native}\t{score}\n")

    print(f"Wrote {len(accumulated)} pairs → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
