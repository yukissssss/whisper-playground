"""
リアルタイム日本語文字起こしツール – medium モデル版
外部 VAD + 音量フィルターでノイズ・フィラーをカット
"""
from postprocess import post_process
import os
import queue
import threading
from typing import List

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

# ===== 基本設定 =====
SAMPLE_RATE = 16_000
FRAME_MS    = 30
CHUNK_MS    = 2_000
BYTES_PER_FRAME = SAMPLE_RATE * 2 * FRAME_MS // 1000
LANG = "ja"

# ===== モデルパラメータ =====
BEAM_SIZE = 8
BEST_OF   = 2
TEMP      = 0.0

# ===== ノイズ & VAD =====
MIN_LEVEL = 3_000          # これ未満の入力は無視
VAD_LEVEL = 0              # 0 = ゆるい

FILLER = {
    "ご視聴ありがとうございました",
    "最後まで視聴してくださって ありがとうございます",
    "ご視聴ありがとうございました。",
}

# ===== モデルと検出器 =====
model = WhisperModel("medium", device="cpu", compute_type="int8")
vad   = webrtcvad.Vad(VAD_LEVEL)

# ===== 音声バッファ =====
q: "queue.Queue[bytes]" = queue.Queue()


def audio_cb(indata: np.ndarray, frames: int, time, status) -> None:
    """小さすぎる入力は捨てる"""
    if np.abs(indata).mean() * 32768 < MIN_LEVEL:
        return
    q.put(bytes(indata))


# ===== 文字起こしスレッド =====
def transcriber() -> None:
    buf = b""
    last_line = ""
    while True:
        buf += q.get()
        while len(buf) >= BYTES_PER_FRAME:
            frame, buf = buf[:BYTES_PER_FRAME], buf[BYTES_PER_FRAME:]

            if not vad.is_speech(frame, SAMPLE_RATE):
                continue

            # ---- CHUNK_MS ミリ秒ぶん集める ----
            speech: List[bytes] = [frame]
            while len(speech) * FRAME_MS < CHUNK_MS:
                if len(buf) < BYTES_PER_FRAME:
                    buf += q.get()
                nxt, buf = buf[:BYTES_PER_FRAME], buf[BYTES_PER_FRAME:]
                speech.append(nxt)

            # ---- 文字起こし ----
            samples = (
                np.frombuffer(b"".join(speech), np.int16)
                .astype(np.float32)
                / 32768.0
            )
            segments, _ = model.transcribe(
                samples,
                language=LANG,
                beam_size=BEAM_SIZE,
                best_of=BEST_OF,
                temperature=TEMP,
            )
            for seg in segments:
                line = seg.text.strip()
                if line in FILLER or line == last_line:
                    continue
                last_line = line
                print(line, flush=True)


# ===== マイク入力ループ =====
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


# ===== エントリーポイント =====
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
            txt = s.text.strip()
            if txt not in FILLER:
                print("ファイル結果:", txt)
    else:
        realtime_caption()
