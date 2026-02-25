import pandas as pd
import sys
import unicodedata
import re

_trans_table = dict.fromkeys(
    c for c in range(sys.maxunicode)
    if unicodedata.category(chr(c)) == 'Mn'
)

def normalize_text(s: str) -> str:
    if not s:
        return ""

    if pd.isna(s):
        return s

    s = str(s).lower()
    s = unicodedata.normalize("NFD", s).translate(_trans_table)
    s = re.sub(r"[^-0-9a-z.\s]", " ", s)
    s = re.sub(r"[-.]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()
