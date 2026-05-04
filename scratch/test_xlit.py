import torch
import argparse
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([argparse.Namespace])

import sys
from unittest.mock import MagicMock
if 'urduhack' not in sys.modules:
    mock_urdu = MagicMock()
    mock_urdu.normalize = lambda x: x
    sys.modules['urduhack'] = mock_urdu

from ai4bharat.transliteration import XlitEngine

print("Loading engine...")
engine = XlitEngine(src_script_type="roman", beam_width=10, rescore=True)
print("Engine loaded.")

try:
    word = "aaj"
    print(f"Testing translit_word for '{word}'...")
    res = engine.translit_word(word, lang_code="hi")
    print(f"Result type: {type(res)}")
    print(f"Result: {res}")
except Exception as e:
    print(f"ERROR: {e}")
