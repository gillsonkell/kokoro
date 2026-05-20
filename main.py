import configparser
import re
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from misaki import en, espeak

script_dir = Path(__file__).resolve().parent
config = configparser.ConfigParser()
config.read(script_dir / "settings.ini")

DEBUG = config.getboolean("settings", "DEBUG")
VOICE = config.get("settings", "VOICE")
BRITISH = config.getboolean("settings", "BRITISH")
NEWLINE_PAUSE = config.getfloat("settings", "NEWLINE_PAUSE")
SENTENCE_PAUSE = config.getfloat("settings", "SENTENCE_PAUSE")
ARTIST = config.get("settings", "ARTIST")
BOOK_TITLE = config.get("settings", "BOOK_TITLE")

PHONEME_LIMIT = 480
SAMPLE_RATE = 24000

g2p = en.G2P(
    british=BRITISH,
    fallback=espeak.EspeakFallback(british=BRITISH)
)
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

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

def build_tasks(text):
    tasks = []
    total_chunks = 0
    for segment in re.split(r"(\n+)", text):
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
    return tasks, total_chunks

def synthesize(tasks, total_chunks):
    chunks = []
    chunks_generated = 0
    for task in tasks:
        if task[0] == "speak":
            phonemes = phonemize(task[1])
            if phonemes:
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
    return np.concatenate(chunks)

def process_file(input_file, output_file, track_number):
    main_start = time.perf_counter()
    text = input_file.read_text(encoding="utf-8").strip()

    tasks, total_chunks = build_tasks(text)

    if DEBUG:
        print("--- Audio Commands ---")
        for task in tasks:
            print(f"pause: {task[1]:g}s" if task[0] == "pause" else f"speak: {task[1]}")
        print("----------------------")

    print("Generating audio...")
    start = time.perf_counter()
    audio = synthesize(tasks, total_chunks)

    raw_output_path = output_file.with_suffix(".raw.wav")
    sf.write(str(raw_output_path), audio, SAMPLE_RATE)
    print(f"Generated in {format_time(time.perf_counter() - start)}")

    print("Post-processing...")
    start = time.perf_counter()
    result = subprocess.run([
        "ffmpeg",
        "-i", str(raw_output_path),
        "-id3v2_version", "3",
        "-write_id3v1", "1",
        "-metadata", f"artist={ARTIST}",
        "-metadata", f"album={BOOK_TITLE}",
        "-metadata", f"title=Chapter {track_number}",
        "-metadata", f"track={track_number}",
        "-af", "loudnorm=I=-19:TP=-3:LRA=11",
        "-ar", str(SAMPLE_RATE),
        "-c:a", "libmp3lame",
        "-q:a", "0",
        "-y",
        str(output_file)
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    raw_output_path.unlink(missing_ok=True)
    print(f"Post-processed in {format_time(time.perf_counter() - start)}")
    print(f"{output_file.name} created in {format_time(time.perf_counter() - main_start)}")

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.3f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"

input_dir = script_dir / "input"
output_dir = script_dir / "output"

input_files = sorted(input_dir.glob("*.txt"))
print(f"Processing {len(input_files)} files.")
global_start = time.perf_counter()
for i, input_file in enumerate(input_files):
    output_file = output_dir / (input_file.stem + ".mp3")
    print(f"\nProcessing {input_file.name}")
    process_file(input_file, output_file, i + 1)

print(f"\nAll files processed in {format_time(time.perf_counter() - global_start)}")
