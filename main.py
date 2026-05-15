import re
import subprocess
import sys
import time
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from misaki import en, espeak

DEBUG = True
VOICE = "bm_lewis"
BRITISH = True
NEWLINE_PAUSE = .5
SENTENCE_PAUSE = .7
PHONEME_LIMIT = 480
SAMPLE_RATE = 24000

g2p = en.G2P(
    british=BRITISH,
    fallback=espeak.EspeakFallback(british=BRITISH)
)

def phonemize(text):
    return g2p(text)[0]

def sentences(paragraph):
    sentence_split_re = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
    for sentence in (s.strip() for s in sentence_split_re.split(paragraph) if s.strip()):
        yield sentence

def pack(sentences):
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
            continue
        candidate = f"{current} {sentence}"
        if len(phonemize(candidate)) <= PHONEME_LIMIT:
            current = candidate
        else:
            yield current
            current = sentence
    if current:
        yield current

chapter_text = open("input_chapters_text/01.txt", encoding="utf-8").read().strip()

tasks = []
total_chunks = 0
for segment in re.split(r"(\n+)", chapter_text):
    if not segment:
        continue
    if segment[0] == "\n":
        tasks.append(("pause", len(segment) * NEWLINE_PAUSE))
        continue
    sentence_packs = list(pack(sentences(segment)))
    total_chunks += len(sentence_packs)
    for i, sentence_pack in enumerate(sentence_packs):
        tasks.append(("speak", sentence_pack))
        if i < len(sentence_packs) - 1:
            tasks.append(("pause", SENTENCE_PAUSE))

if DEBUG:
    print("--- Audio Commands ---")
    for task in tasks:
        print(f"pause: {task[1]:g}s" if task[0] == "pause" else f"speak: {task[1]}")
    print("----------------------")

print("Generating audio...")
start = time.perf_counter()
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
chunks = []
chunks_generated = 0
for task in tasks:
    if task[0] == "speak":
        samples, _ = kokoro.create(
            phonemize(task[1]),
            is_phonemes=True,
            voice=VOICE,
            speed=0.9,
        )
        chunks.append(samples)
        chunks_generated += 1
        print(f"Generated chunk {chunks_generated}/{total_chunks}")
    else:
        chunks.append(
            np.zeros(
                int(task[1] * SAMPLE_RATE),
                dtype=np.float32,
            )
        )
raw_output_path = "output_raw.wav"
sf.write(raw_output_path, np.concatenate(chunks), SAMPLE_RATE)
print(f"Generated in {time.perf_counter() - start:.3f}s.")

print("Post-processing...")
start = time.perf_counter()
result = subprocess.run([
    "ffmpeg", "-i", raw_output_path,
    "-af", "loudnorm=I=-19:TP=-3:LRA=11",
    "-ar", str(SAMPLE_RATE),
    "-y",
    "output.wav"
], capture_output=True, text=True)
if result.returncode != 0:
    print(f"FFmpeg failed:\n{result.stderr}", file=sys.stderr)
    sys.exit(1)
print(f"Post-processed in {time.perf_counter() - start:.3f}s.")

print("Finished.")
