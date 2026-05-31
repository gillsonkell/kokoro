import configparser
import re
import sys
import time
from pathlib import Path

import httpx

script_dir = Path(__file__).resolve().parent
config = configparser.ConfigParser()
config.read(script_dir / "settings.ini")

OPENROUTER_API_KEY = config.get("settings", "OPENROUTER_API_KEY")
OPENROUTER_MODEL = config.get("settings", "OPENROUTER_MODEL")

DEBUG = config.getboolean("settings", "DEBUG")
PROCESS_INPUT_DIR = script_dir / "preprocess_input"
PROCESS_OUTPUT_DIR = script_dir / "preprocess_output"
PROCESS_INSTRUCTIONS_FILE = script_dir / "preprocess_instructions.txt"
SENTENCE_SPLIT = re.compile(r"((?<=[.!?])\s+(?=[A-Z\"'(]))")

MAX_CHARS = 2000


def load_instructions():
    return PROCESS_INSTRUCTIONS_FILE.read_text(encoding="utf-8")


def chunk_text(text):
    parts = SENTENCE_SPLIT.split(text)

    chunks, current = [], ""
    for part in parts:
        if current and len(current) + len(part) > MAX_CHARS:
            chunks.append(current)
            current = part
        else:
            current += part

    if current:
        chunks.append(current)

    return chunks


def preprocess_chunk(chunk, instructions):
    system_prompt = f"""You are a text preprocessing tool for text-to-speech.
When instructed to use specific phonemes, use this format: [Word](/wˈɜɹd/). Don't forget the square brackets, parentheses, or forward slashes.
Apply the following rules to the text. Only output the corrected text, nothing else.
Preserve all whitespace exactly: keep every space and line break where it is. Do not merge lines, add or remove blank lines, or re-wrap text.
Do not add quotes, explanations, or any formatting. Just return the corrected text.

Rules:
{instructions}"""

    with httpx.Client(timeout=300) as client:
        response = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk},
                ],
            },
        )
        response.raise_for_status()
        data = response.json()
        response_text = data["choices"][0]["message"]["content"]

        # OpenRouter reports the actual USD cost of the request in usage.cost.
        # It may be absent (e.g. free models) so default to 0.0.
        cost = (data.get("usage") or {}).get("cost") or 0.0

        if DEBUG:
            print(f"\nSent: {chunk}")
            print(f"Received: {response_text}\n")

        return response_text, cost


def process_file(input_path, instructions):
    text = input_path.read_text(encoding="utf-8")

    # Split on newline runs, capturing them (odd indices) so the exact newline
    # layout is restored verbatim. The model never sees a newline.
    pieces = re.split(r"(\n+)", text)

    # Pre-compute the segments to send so progress shows an accurate total.
    # (text, send): send=True means call the model; False means emit verbatim
    # (newline runs and blank/whitespace-only pieces).
    segments = []
    for j, piece in enumerate(pieces):
        if j % 2 == 1 or not piece.strip():
            segments.append((piece, False))
        else:
            for sub in chunk_text(piece):  # piece has no newlines, so neither do its chunks
                segments.append((sub, bool(sub.strip())))

    total = sum(1 for _, send in segments if send)

    results, n, file_cost = [], 0, 0.0
    for seg, send in segments:
        if not send:
            results.append(seg)
            continue

        lead, core, trail = re.match(r"(?s)(\s*)(.*?)(\s*)$", seg).groups()

        n += 1
        start = time.perf_counter()
        processed, cost = preprocess_chunk(core, instructions)
        file_cost += cost
        # This segment had no newlines; drop any the model added.
        processed = re.sub(r"\s*\n+\s*", " ", processed).strip()
        print(f"Generated chunk {n}/{total} in {format_time(time.perf_counter() - start)} ({len(core)} chars sent, {len(processed)} received, {format_cost(cost)})")
        results.append(f"{lead}{processed}{trail}")

    return "".join(results), file_cost


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.3f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


def format_cost(dollars):
    return f"${dollars:.6f}"


instructions = load_instructions()

input_files = sorted(PROCESS_INPUT_DIR.glob("*.txt"))
if not input_files:
    print(f"No .txt files found in {PROCESS_INPUT_DIR}")
    sys.exit(1)

PROCESS_OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Processing {len(input_files)} files.")
global_start = time.perf_counter()

total_cost = 0.0
for input_file in input_files:
    print(f"\nProcessing {input_file.name}")
    start = time.perf_counter()
    output_file = PROCESS_OUTPUT_DIR / input_file.name
    result, file_cost = process_file(input_file, instructions)
    total_cost += file_cost
    output_file.write_text(result, encoding="utf-8")
    print(f"{output_file.name} created in {format_time(time.perf_counter() - start)} (cost: {format_cost(file_cost)})")

print(f"\nAll files processed in {format_time(time.perf_counter() - global_start)} (total cost: {format_cost(total_cost)})")