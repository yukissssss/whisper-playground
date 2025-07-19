"""
リアルタイム日本語文字起こしツール – medium モデル版（2025‑07‑17）
-----------------------------------------------------------------
* `--wait_timeout_ms` で **固定待機タイムアウト** を CLI 指定
* 引数が無ければ従来の `CHUNK_MS` ぶん収集（動的待機ロジック）
* デバッグ用に `seg/s` と `timeout` を stderr へ出力
"""

from __future__ import annotations

# ── 標準ライブラリ ─────────────────────────
import argparse
import os
import queue
import sys
import threading
import time
from typing import List

# ── 外部ライブラリ ─────────────────────────
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from postprocess import post_process  # 後処理パイプライン

# ===== CLI =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--wait_timeout_ms", type=int, default=None,
                    help="無音復帰までの固定待機時間（ms）")
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                    help="推論デバイス")
parser.add_argument("--compute_type", default="int8",
                    choices=["int8", "int16", "float16", "float32"],
                    help="量子化精度")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

# ===== 基本設定 =================================================
SAMPLE_RATE = 16_000            # 16 kHz
FRAME_MS    = 30                # VAD 判定フレーム長
CHUNK_MS    = 2_000             # 従来の最大チャンク
WAIT_TIMEOUT_MS: int | None = args.wait_timeout_ms
if args.debug:
    print(f"DEBUG timeout = {WAIT_TIMEOUT_MS}", file=sys.stderr)

BYTES_PER_FRAME = SAMPLE_RATE * 2 * FRAME_MS // 1000  # 16‑bit mono
LANG = "ja"

# ===== モデルパラメータ ========================================
BEAM_SIZE = 8
BEST_OF   = 2
TEMP      = 0.0

# ===== ノイズ/VAD 閾値 =========================================
MIN_LEVEL = 3_000  # 入力振幅がこれ未満なら遮断
VAD_LEVEL = 0      # 0=ゆるい, 3=厳格

FILLER = {
    "ご視聴ありがとうございました",
    "最後まで視聴してくださって ありがとうございます",
    "ご視聴ありがとうございました。",
}

# ===== モデルロード ============================================
model = WhisperModel("medium", device=args.device, compute_type=args.compute_type)
vad   = webrtcvad.Vad(VAD_LEVEL)

# ===== 音声バッファ ============================================
aud_q: "queue.Queue[bytes]" = queue.Queue()


def audio_cb(indata: np.ndarray, frames: int, _time, _status) -> None:
    """PortAudio コールバック: ノイズフィルタ＆バッファ投入"""
    if np.abs(indata).mean() * 32768 < MIN_LEVEL:
        return
    aud_q.put(bytes(indata))


# ===== 文字起こしスレッド =====================================

def transcriber() -> None:
    buf = b""
    last_line = ""
    processed_frames = 0
    start_time = time.time()

    while True:
        buf += aud_q.get()
        while len(buf) >= BYTES_PER_FRAME:
            frame, buf = buf[:BYTES_PER_FRAME], buf[BYTES_PER_FRAME:]

            if not vad.is_speech(frame, SAMPLE_RATE):
                continue

            # ---- スピーチ塊を収集 ----
            speech: List[bytes] = [frame]
            silence_ms = 0
            # 音声が続く限りフレームを取得、連続無音が WAIT_TIMEOUT_MS 以上で break
            limit_ms = WAIT_TIMEOUT_MS or CHUNK_MS
            while True:
                if len(buf) < BYTES_PER_FRAME:
                    buf += aud_q.get()
                nxt, buf = buf[:BYTES_PER_FRAME], buf[BYTES_PER_FRAME:]
                speech.append(nxt)

                if vad.is_speech(nxt, SAMPLE_RATE):
                    silence_ms = 0  # 音声が来たらリセット
                else:
                    silence_ms += FRAME_MS

                # 終了条件: 無音が一定時間続いた もしくは CHUNK_MS 上限
                if (silence_ms >= limit_ms) or (len(speech) * FRAME_MS >= CHUNK_MS):
                    break

            # ---- Whisper 推論 ----
            samples = (
                np.frombuffer(b"".join(speech), np.int16).astype(np.float32) / 32768.0
            )
            segments, _ = model.transcribe(
                samples,
                language=LANG,
                beam_size=BEAM_SIZE,
                best_of=BEST_OF,
                temperature=TEMP,
            )
            for seg in segments:
                raw = seg.text.strip()
                line = post_process(raw)
                if line in FILLER or line == last_line:
                    continue
                last_line = line
                print(line, flush=True)

            # ---- デバッグ: 処理速度 ----
            processed_frames += len(speech)
            if args.debug and processed_frames >= SAMPLE_RATE:  # 毎秒表示
                elapsed = time.time() - start_time
                print(f"seg/s = {processed_frames/elapsed:.2f}", file=sys.stderr)
                processed_frames = 0
                start_time = time.time()


# ===== マイク入力 ==============================================

def realtime_caption() -> None:
    threading.Thread(target=transcriber, daemon=True).start()
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        dtype="int16",
        channels=1,
        blocksize=0,
        callback=audio_cb,
    ):
        print("🎙️  話してください (Ctrl+C で終了)")
        while True:
            sd.sleep(1_000)


# ===== エントリーポイント =====================================
if __name__ == "__main__":
    if os.path.exists("test.wav"):
        import soundfile as sf

        samples, _ = sf.read("test.wav", dtype="float32")
        segs, _ = model.transcribe(
            samples,
            language=LANG,
            beam_size=BEAM_SIZE,
            best_of=BEST_OF,
            temperature=TEMP,
        )
        for s in segs:
            raw = s.text.strip()
            txt = post_process(raw)
            if txt not in FILLER:
                print("ファイル結果:", txt)
    else:
        realtime_caption()
