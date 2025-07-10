# realtime.py  ← コピペで作る
import queue, sounddevice as sd
from faster_whisper import WhisperModel
from silero import VAD

SAMPLE_RATE = 16000
model = WhisperModel("small-int8", device="cpu", compute_type="int8")
vad = VAD(sample_rate=SAMPLE_RATE)
q = queue.Queue()

def cb(indata, frames, time, status):
    q.put(bytes(indata))

def loop():
    buf = b""
    while True:
        buf += q.get()
        if len(buf) > SAMPLE_RATE*2:        # 1秒たまったら
            if vad.is_speech(buf):
                txt, _ = model.transcribe(buf, beam_size=1)
                print(txt, flush=True)
            buf = b""

import threading, sys
threading.Thread(target=loop, daemon=True).start()
with sd.RawInputStream(samplerate=SAMPLE_RATE, dtype='int16',
                       channels=1, callback=cb):
    print("🎙️  話してください (Ctrl+C で終了)")
    while True: sd.sleep(1000)
