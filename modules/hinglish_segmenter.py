# modules/hinglish_segmenter.py
# ─────────────────────────────────────────────────────────────────────────────
# Splits romanized Hinglish into ordered [{lang, text}] segments.
# Vocabulary-based, no ML, near-zero latency.
# Unknown words default to Hindi.
# ─────────────────────────────────────────────────────────────────────────────

import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

HINDI_WORDS = {
    "main","mujhe","mujhko","mera","meri","mere","hum","humara","humari",
    "humare","tum","tumhara","tumhari","tumhare","aap","aapka","aapki",
    "aapke","vo","woh","unka","unki","unke","iska","iski","iske","inhe",
    "use","usse","hai","hain","tha","thi","the","hoga","hogi","honge",
    "ho","hua","hui","hue","karo","karna","kar","kiya","ki","jaana",
    "jao","gaya","gayi","gaye","aa","aao","aana","aaya","aayi","aayen",
    "le","lena","liya","de","dena","diya","bol","bolna","bola","boli",
    "dekho","dekhna","dekha","dekhi","suno","sunna","suna","bata","batao",
    "batana","bataya","samjho","samajh","chahiye","sakta","sakti","sakte",
    "raha","rahi","rahe","rhe","chahta","chahti","aaj","kal","parso",
    "abhi","phir","jab","tab","kab","yahan","wahan","kahan","idhar",
    "udhar","kyun","kaise","kya","kaun","kitna","kitni","kitne","bahut",
    "thoda","zyada","kam","bilkul","sirf","bhi","aur","ya","lekin",
    "magar","kyunki","isliye","toh","to","na","nahi","haan","achha",
    "theek","sahi","galat","naya","purana","bada","chota","accha","bura",
    "sundar","mushkil","aasaan","yaar","bhai","dost","ji","namaste",
    "kaam","din","raat","subah","sham","ghanta","mahina","saal","paise",
    "rupaye","ghar","daftar","raasta","khana","paani","chai","roti",
    "naam","baat","cheez","jagah","taraf","waqt","baar","tarah","log",
    "mein","se","ko","ke","par","pe","tak","liye","saath","baad",
    "pehle","ander","bahar","ek","do","teen","char","paanch","chhe",
    "saat","aath","nau","das","bees","tees","sau","hazaar",
    "meeting","office","phone","laptop","email","internet",  # kept in Hindi too
}

ENGLISH_WORDS = {
    "meeting","call","email","phone","laptop","computer","internet","wifi",
    "password","download","upload","file","folder","project","deadline",
    "presentation","report","document","office","team","manager","boss",
    "client","server","database","software","app","update","install",
    "login","account","profile","settings","check","fix","send","share",
    "open","close","start","stop","okay","ok","yes","no","thanks","hello",
    "hi","bye","good","bad","nice","great","perfect","sure","right",
    "wrong","problem","issue","solution","idea","plan","at","in","on",
    "for","with","from","by","and","or","but","the","is","are","will",
    "can","should","am","pm","today","tomorrow","week","month","year",
    "date","time","schedule","monday","tuesday","wednesday","thursday",
    "friday","saturday","sunday","percent","urgent","ready","done",
    "approve","approved","dispatch","deliver","review","feedback",
    "conference","room","free","slow","charge","connect","disconnect",
}


def segment_hinglish(text: str) -> List[Dict]:
    """Split romanized Hinglish into [{lang:'hi'|'en', text:'...'}] segments."""
    if not text or not text.strip():
        return []

    text = re.sub(r'\s+', ' ', text.strip())
    tokens = text.split()
    word_langs = []

    for token in tokens:
        clean = re.sub(r"[^\w'-]", "", token).lower()
        if not clean:
            if word_langs:
                word_langs[-1] = (word_langs[-1][0] + " " + token, word_langs[-1][1])
            continue
        if re.match(r'^\d+$', clean):
            word_langs.append((token, "num"))
            continue
        lang = "hi" if clean in HINDI_WORDS else ("en" if clean in ENGLISH_WORDS else "hi")
        word_langs.append((token, lang))

    # Resolve numbers — attach to language of following word
    resolved = []
    pending_nums = []
    for word, lang in word_langs:
        if lang == "num":
            pending_nums.append(word)
        else:
            if pending_nums:
                resolved.append((" ".join(pending_nums), lang))
                pending_nums = []
            resolved.append((word, lang))
    if pending_nums:
        last_lang = resolved[-1][1] if resolved else "hi"
        resolved.append((" ".join(pending_nums), last_lang))

    # Merge consecutive same-language tokens
    segments = []
    for word, lang in resolved:
        if segments and segments[-1]["lang"] == lang:
            segments[-1]["text"] += " " + word
        else:
            segments.append({"lang": lang, "text": word.strip()})

    return [{"lang": s["lang"], "text": s["text"].strip()} for s in segments if s["text"].strip()]
