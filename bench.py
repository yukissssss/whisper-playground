from faster_whisper import WhisperModel
from postprocess import post_process
import time

AUDIO = "test.wav"
model = WhisperModel("medium", device="cpu", compute_type="int8")

t0 = time.time()
segments, info = model.transcribe(AUDIO, beam_size=8, best_of=2)

for s in segments:
    print(post_process(s.text))

elapsed = time.time() - t0
print(f"\n音声長 {info.duration:.1f}s / 処理時間 {elapsed:.1f}s "
      f"=> seg/s {info.duration/elapsed:.2f}")

