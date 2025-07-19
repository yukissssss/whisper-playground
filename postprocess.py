#!/usr/bin/env python3
"""postprocess.py — Whisper 出力を整形して誤変換を補正

■ 概要
Whisper の日本語文字起こし結果を医療用途向けに後処理するスクリプト。
置換ルールは **外部 TSV 辞書**（デフォルト: `med_dict.tsv`）から読み込む。

TSV 形式: pattern<TAB>replace<TAB>is_regex(0/1)<TAB>note(任意)
- is_regex が 0 → リテラル置換（str.replace）
- is_regex が 1 → 正規表現置換（re.sub, flags=re.I）

辞書が見つからない場合は、スクリプト内の FALLBACK_DICT を使用する。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Tuple

import neologdn

# ----------------------------------------------------------------------
# Fallback dictionary (ユーザー提供ルール)
# ----------------------------------------------------------------------
FALLBACK_DICT: List[Tuple[str, str, int]] = [
    (r"手素", "主訴", 1),
    (r"三素", "主訴", 1),
    (r"スポ.?2", "SpO2", 1),
    (r"スポー?に?", "SpO2", 1),
    (r"SpO2\\s*(\\d+)", r"SpO2 \\1", 1),
    (r"c ?r ?p", "CRP", 1),
    (r"c\\s*ガール\\s*p\\s*12\\s*(?:[\\.]|ピリオド)?\\s*3", "CRP12.3", 1),
    (r"レバーフロキサ.*", "レボフロキサシン", 1),
    (r"ナイク", "内科", 1),
    (r"若白", "脈拍", 1),
    (r"貼って9?1200", "白血球1万1200", 1),
    (r"4リットル", "毎デシリットル", 1),
    (r"土石", "と咳", 1),
    (r"ミリグラムごとで", "ミリグラム毎", 1),
    (r"c\\s*d", "CT", 1),
    (r"右下歯", "右下葉", 1),
    (r"(\\d+)の(\\d+)", r"\\1/\\2", 1),
    (r"ミリグラム毎毎", "ミリグラム毎", 1),
    (r"ミリグラム毎デシリットル", "mg/dL", 1),
    (r"CRP\\s*([\\d\\.]+)\\s*ミリグラム毎デシリットル", r"CRP \\1 mg/dL", 1),
    (r"CRP\\s*([\\d\\.]+)\\s*mg/?dL", r"CRP \\1 mg/dL", 1),
    (r"(\\d+)ミリグラム", r"\\1 mg", 1),
    (r"(\\d+)\\s*mg\\b", r"\\1 mg", 1),
    (r"(\\d+)ミリ", r"\\1 mm", 1),
    (r"(\\d+)\\s*パーセント", r"\\1 %", 1),
    (r"(\\d+)%", r"\\1 %", 1),
    (r"(\\d+)％", r"\\1 %", 1),
    (r"SpO2\\s*(\\d+)\\s*%", r"SpO2 \\1 %", 1),
]

# ----------------------------------------------------------------------
# TSV 辞書ロード
# ----------------------------------------------------------------------

def load_tsv_dict(path: Path) -> List[Tuple[str, str, int]]:
    if not path.exists():
        return FALLBACK_DICT
    rules: List[Tuple[str, str, int]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            pattern, repl, is_rx, *_ = row + ["", "0"]
            rules.append((pattern, repl, int(is_rx)))
    return rules or FALLBACK_DICT

# ----------------------------------------------------------------------
# 正規化ユーティリティ
# ----------------------------------------------------------------------
_unit_gap = re.compile(r"\s+(μm|mg/dL|mmHg|℃|％|%)", re.I)

_UNIT_PATTERNS = [
    (r"(\d+)\s*mmhg", r"\1 mmHg"),
    (r"(\d+)\s*mg/?dl", r"\1 mg/dL"),
    (r"(\d+)\s*mm\\b", r"\1 mm"),
    (r"(\d+)\s*μm\\b", r"\1 μm"),
    (r"(\d+)\s*％", r"\1 %"),
    (r"(\d+)\s*パーセント", r"\1 %"),
    (r"(\d+)\s*%", r"\1 %"),
]

_CJK_LATIN = re.compile(
    r"([\u4E00-\u9FFF\u3040-\u30FF])([A-Za-z0-9])|([A-Za-z0-9])([\u4E00-\u9FFF\u3040-\u30FF])"
)

_POST_COMMA_PAT = re.compile(r"(mg/dL|mmHg|%|％)(\S)", re.I)


def _normalize_units(text: str) -> str:
    for pat, repl in _UNIT_PATTERNS:
        text = re.sub(pat, repl, text, flags=re.I)
    return text


def _insert_period(text: str) -> str:
    if len(text) > 30 and not text.endswith(("。", ".", "！", "？")):
        text += "。"
    return text

# ----------------------------------------------------------------------
# メイン後処理関数
# ----------------------------------------------------------------------

def post_process(text: str, rules: List[Tuple[str, str, int]]) -> str:
    text = neologdn.normalize(text)
    for pattern, repl, is_rx in rules:
        text = re.sub(pattern, repl, text, flags=re.I) if is_rx else text.replace(pattern, repl)

    text = unicodedata.normalize("NFKC", text)
    text = _unit_gap.sub(r"\1", text)
    text = _normalize_units(text)

    # CJK と英数字の間にスペースを強制挿入
    text = _CJK_LATIN.sub(lambda m: (m.group(1) or m.group(3)) + " " + (m.group(2) or m.group(4)), text)

    text = _POST_COMMA_PAT.sub(r"\1、\2", text)
    text = _insert_period(text)
    return re.sub(r"\s+", " ", text).strip()

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Whisper post‑processing script")
    p.add_argument("input", nargs="?", type=Path, help="入力テキストファイル (省略で stdin)")
    p.add_argument("-o", "--output", type=Path, help="出力ファイル (省略で stdout)")
    p.add_argument("-d", "--dict", type=Path, default=Path("med_dict.tsv"), help="TSV 辞書ファイル")
    return p


def main(argv: List[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    rules = load_tsv_dict(args.dict)
    raw_text = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
    processed = post_process(raw_text, rules)
    if args.output:
        args.output.write_text(processed, encoding="utf-8")
    else:
        print(processed)


if __name__ == "__main__":
    main()
