import configparser
import shutil
import subprocess
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent

config = configparser.ConfigParser()
config.read(script_dir / "settings.ini")

ARTIST = config.get("settings", "ARTIST")
BOOK_TITLE = config.get("settings", "BOOK_TITLE")

output_dir = script_dir / "output"
output_files = sorted(output_dir.glob("*.mp3"))

for i, output_file in enumerate(output_files):
    print(f"Processing: {output_file.name}")
    temp_file = output_file.with_suffix(".tmp.mp3")
    result = subprocess.run([
        "ffmpeg",
        "-i", str(output_file),
        "-c:a", "copy",
        "-id3v2_version", "3",
        "-write_id3v1", "1",
        "-metadata", f"artist={ARTIST}",
        "-metadata", f"album={BOOK_TITLE}",
        "-metadata", f"title=Chapter {i + 1}",
        "-metadata", f"track={i + 1}",
        "-y",
        str(temp_file)
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    shutil.move(temp_file, output_file)

print("All done")
