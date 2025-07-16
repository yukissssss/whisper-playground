# postprocess.py --- Whisper 出力を整形して誤変換を補正
import re, unicodedata, neologdn

MED_DICT = {
    # ── 基本補正 ─────────────────────
    r"手素": "主訴",
    r"三素": "主訴",
    r"スポ.?2": "SpO2",
    r"スポー?に?": "SpO2",                   # スポーに → SpO2
    r"SpO2\s*(\d+)": r"SpO2 \1",             # SpO295% → SpO2 95%
    r"c ?r ?p": "CRP",

    # CRP12.3 の崩れ（全角空白・ピリオド表記・数字の前後ゆれを許容）
    r"c\s*ガール\s*p\s*12\s*(?:[\.]|ピリオド)?\s*3": "CRP12.3",

    # 薬剤ほか
    r"レバーフロキサ.*": "レボフロキサシン",

    # 診療科・測定値
    r"ナイク": "内科",
    r"若白": "脈拍",

    # 数値＆単位まわり
    r"貼って9?1200": "白血球1万1200",
    r"4リットル": "毎デシリットル",

    # ── ★ 追加補正 ──────────────────
    r"土石": "と咳",                             # 発熱土石 → 発熱と咳
    r"ミリグラムごとで": "ミリグラム毎",         # …ごとで毎 → …毎
    r"c\s*d": "CT",                             # 胸部c d → 胸部CT
    r"右下歯": "右下葉",                         # 右下歯 → 右下葉
    r"(\d+)の(\d+)": r"\1/\2",                   # 142の86 → 142/86
    r"ミリグラム毎毎": "ミリグラム毎"            # 毎毎 → 毎
}

_unit_gap = re.compile(r"\s+(mg|mm|℃|％|パーセント)", re.I)  # 

def post_process(text: str) -> str:
    text = neologdn.normalize(text)
    for pat, repl in MED_DICT.items():
        text = re.sub(pat, repl, text, flags=re.I)
    text = unicodedata.normalize("NFKC", text)
    text = _unit_gap.sub(r"\1", text)
    return re.sub(r'\s+', ' ', text).strip()

