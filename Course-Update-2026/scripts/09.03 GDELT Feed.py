"""CS-777 Chapter 9 — the feeder that turns GDELT into a stream.

GDELT (https://www.gdeltproject.org/) extracts world events from news media and
publishes a new export every 15 minutes. This is an ordinary Python program, not a
Spark application: it polls GDELT's `lastupdate.txt`, downloads the newest export
when one appears, and places the finished CSV in the directory the streaming query
is watching. Run it beside `09.04 GDELT Stream.py`, in its own terminal:

    .venv/bin/python3 "code/scripts/09.03 GDELT Feed.py"

**Files are moved in, never written in place.** The download and the unzip both
happen in a staging directory, and only the completed CSV is moved across. Spark's
file source treats every file appearing in the directory as new data and will read
a half-written one, so an atomic move is what keeps a partial file out of the
stream. The archive is never unpacked into the watched directory for the same
reason: a `.zip` sitting there would be read as if it were a CSV.
"""

import datetime as dt
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile

import requests

LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
EXPORT_RE = re.compile(r"\bhttp://data\.gdeltproject\.org/gdeltv2/(\d{14})\.export\.CSV\.zip\b")
EXPORT_URL = "http://data.gdeltproject.org/gdeltv2/{stamp}.export.CSV.zip"
# lastupdate.txt is sometimes published before its files are uploaded, and a slot
# is occasionally skipped for good. On a 404, step back one 15-minute slot at a
# time, up to an hour, and take the newest export that exists.
FALLBACK_SLOTS = 4

# Relative to wherever the program is started, so the files are easy to find.
INPUT_PATH = os.path.join(os.getcwd(), "gdelt-input")
CHECK_EVERY = 300  # seconds; GDELT publishes a new export every 15 minutes


def place(src: str, input_path: str) -> None:
    """Move a finished file into the watched directory in one step."""
    dest = os.path.join(input_path, os.path.basename(src))
    if os.path.exists(dest):
        print("already placed:", os.path.basename(dest), flush=True)
        return
    # os.replace is atomic within one filesystem, so the streaming query never
    # sees a partially written file.
    os.replace(src, dest)
    print("placed", dest, flush=True)


def fetch_latest(input_path: str, staging: str) -> None:
    """Download the newest export, unzip it in staging, then move the CSV in."""
    try:
        response = requests.get(LASTUPDATE_URL, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print("Error fetching last update info:", exc, flush=True)
        return

    match = EXPORT_RE.search(response.text.strip())
    if not match:
        print("no export URL in lastupdate.txt", flush=True)
        return

    latest = dt.datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    for step in range(FALLBACK_SLOTS + 1):
        stamp = (latest - dt.timedelta(minutes=15 * step)).strftime("%Y%m%d%H%M%S")
        csv_url = EXPORT_URL.format(stamp=stamp)
        archive = os.path.join(staging, csv_url.split("/")[-1])
        expected_csv = os.path.join(input_path, os.path.basename(archive)[: -len(".zip")])
        if os.path.exists(expected_csv):
            print("already placed:", os.path.basename(expected_csv), flush=True)
            return
        try:
            with requests.get(csv_url, stream=True, timeout=120) as response:
                if response.status_code == 404:
                    print("not published yet:", os.path.basename(archive), flush=True)
                    continue
                response.raise_for_status()
                with open(archive, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=65536):
                        handle.write(chunk)
        except requests.exceptions.RequestException as exc:
            print("Error downloading the file:", exc, flush=True)
            return
        break
    else:
        print("no export available in the last hour", flush=True)
        return
    print("downloaded", os.path.basename(archive), flush=True)

    try:
        with zipfile.ZipFile(archive) as archive_file:
            archive_file.extractall(staging)
    except zipfile.BadZipFile as exc:
        print("Error unzipping the file:", exc, flush=True)
        return
    finally:
        os.remove(archive)

    for name in sorted(os.listdir(staging)):
        if name.endswith(".CSV"):
            place(os.path.join(staging, name), input_path)


def main() -> int:
    os.makedirs(INPUT_PATH, exist_ok=True)
    print("writing exports to", INPUT_PATH, flush=True)

    staging = tempfile.mkdtemp(prefix="gdelt-staging-")
    try:
        while True:
            fetch_latest(INPUT_PATH, staging)
            print(f"Waiting {CHECK_EVERY} seconds for the next check...", flush=True)
            time.sleep(CHECK_EVERY)
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
