# modules/aksharantar_lexicon.py
# Roman (Aksharantar “english word” field) → Devanagari (“native word”) lookup
# Built offline; runtime only reads the exported file.

import json
import logging
import os
import re
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def normalize_lex_key(word: str) -> str:
    return word.lower().strip(".,!?;:'\"")


def load_aksharantar_lexicon(path: str) -> Dict[str, str]:
    """
    Load roman→Devanagari map from disk.

    Supported:
      - .tsv / .txt: columns roman<TAB>devanagari [TAB optional_score]
      - .json: object { "roman": "देव..." , ... }
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Aksharantar lexicon not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    out: Dict[str, str] = {}

    if ext in (".tsv", ".txt"):
        # Optional third column: higher score wins
        scored: Dict[str, Tuple[float, str]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.rstrip("\n\r")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    parts = [p.strip() for p in re.split(r"[,\|]", line) if p.strip()]
                if len(parts) < 2:
                    logger.debug("Skipping bad line %s: %s", lineno, line[:80])
                    continue
                key = normalize_lex_key(parts[0])
                dev = parts[1].strip()
                if not key or not dev:
                    continue
                score = float(parts[2]) if len(parts) > 2 else 0.0
                prev = scored.get(key)
                if prev is None or score >= prev[0]:
                    scored[key] = (score, dev)
        out = {k: v[1] for k, v in scored.items()}

    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("JSON lexicon must be an object mapping roman→Devanagari")
        for k, v in raw.items():
            nk = normalize_lex_key(str(k))
            if nk:
                out[nk] = str(v).strip()

    else:
        raise ValueError(f"Unsupported lexicon extension {ext}: use .tsv, .txt, or .json")

    logger.info("Loaded Aksharantar-format lexicon: %s (%d entries)", path, len(out))
    return out


def try_load_lexicon(path: Optional[str]) -> Optional[Dict[str, str]]:
    """Return map or None on missing path / error (logs warning)."""
    if not path:
        return None
    try:
        return load_aksharantar_lexicon(path)
    except Exception as e:
        logger.warning("Aksharantar lexicon disabled: %s", e)
        return None
