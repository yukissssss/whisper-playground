"""
リアルタイム日本語文字起こしツール – medium モデル版（2025‑07‑21）
-----------------------------------------------------------------
* 外部 VAD と音量フィルターでノイズ・フィラーをカット
* `--wait_timeout_ms` で **固定待機タイムアウト** を CLI 指定（未指定なら動的）
* `--device` / `--compute_type` で推論デバイス・量子化精度を選択
* `--debug` で `seg/s` と `timeout` を stderr 出力
* 入力ファイル (`test.wav`) があればバッチ処理、無ければマイクでリアルタイム
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
from postprocess import post_process  # 医療用後処理パイプライン

# ===== CLI =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--wait_timeout_ms", type=int, default=None,
                    help="無音復帰までの固定待機時間（ms）。未指定なら動的ロジック")
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                    help="Whisper 推論デバイス")
parser.add_argument("--compute_type", default="int8",
                    choices=["int8", "int16", "float16", "float32"],
                    help="量子化精度 (faster‑whisper)")
parser.add_argument("--debug", action="store_true", help="デバッグ情報を stderr に表示")
parser.add_argument("--input", help="解析する wav ファイル (16kHz mono)。省略でマイク入力")
args = parser.parse_args()

# ===== 基本設定 =================================================
SAMPLE_RATE = 16_000            # 16 kHz mono
FRAME_MS    = 30                # VAD 判定フレーム長 (ms)
CHUNK_MS    = 2_000             # 動的ロジック時の最大チャンク長 (ms)
WAIT_TIMEOUT_MS: int | None = args.wait_timeout_ms  # 固定タイムアウト
if args.debug:
    print(f"DEBUG fixed timeout = {WAIT_TIMEOUT_MS} ms", file=sys.stderr)

BYTES_PER_FRAME = SAMPLE_RATE * 2 * FRAME_MS // 1000  # 16‑bit PCM, mono
LANG = "ja"

# ===== モデルパラメータ ========================================
BEAM_SIZE = 12   # より高精度
BEST_OF   = 7
TEMP      = 0.0

# ===== ノイズ/VAD 閾値 =========================================
MIN_LEVEL = 3_000  # これ未満の入力は無視
VAD_LEVEL = 0      # 0=ゆるい, 3=厳格 (WebRTC VAD)

FILLER = {
    "ご視聴ありがとうございました",
    "最後まで視聴してくださって ありがとうございます",
    "ご視聴ありがとうございました。",
}

# ===== モデル & 検出器 =========================================
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
            limit_ms = WAIT_TIMEOUT_MS or CHUNK_MS  # 動的 or 固定
            while True:
                if len(buf) < BYTES_PER_FRAME:
                    buf += aud_q.get()
                nxt, buf = buf[:BYTES_PER_FRAME], buf[BYTES_PER_FRAME:]
                speech.append(nxt)

                if vad.is_speech(nxt, SAMPLE_RATE):
                    silence_ms = 0
                else:
                    silence_ms += FRAME_MS

                # 終了条件: 無音が一定時間続いた or 最大チャンク長
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
            if args.debug and processed_frames >= SAMPLE_RATE:
                elapsed = time.time() - start_time
                print(f"seg/s = {processed_frames/elapsed:.2f}", file=sys.stderr)
                processed_frames = 0
                start_time = time.time()

# ===== マイク入力ループ ========================================

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
    if args.input and os.path.exists(args.input):
        import soundfile as sf

        samples, _ = sf.read(args.input, dtype="float32")
        segs, _ = model.transcribe(
            samples,
            language=LANG,
            beam_size=BEAM_SIZE,
            best_of=BEST_OF,
            temperature=TEMP,
        )
        for s in segs:
            txt = post_process(s.text.strip())
            if txt not in FILLER:
                print("ファイル結果:", txt)
    else:
        realtime_caption()
