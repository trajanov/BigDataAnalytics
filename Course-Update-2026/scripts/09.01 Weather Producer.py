"""CS-777 Chapter 9 — the producer half of the Kafka weather example.

One JSON sensor reading per second, published to the `weatherdata` topic. This is
an ordinary Python program using an ordinary Kafka client: nothing about it knows
that Spark is on the other end, which is the decoupling the chapter describes.

Start the broker first, then run this in its own terminal:

    code/scripts/kafka/start.sh
    .venv/bin/python3 "code/scripts/09.01 Weather Producer.py"

and watch the messages arrive at http://localhost:8080 (topic `weatherdata`).

The reading's key is the station name. Kafka routes by key, so every reading from
one station lands in one partition and is read back in the order it was written.
That is the whole of the ordering guarantee the chapter states, and it is visible
in Kafka UI.

These four names are not arbitrary. Kafka hashes the key with murmur2 and takes
it modulo the partition count, and over three partitions these four fall 2, 2, 1:

    boston -> 2    denver -> 2    chicago -> 1    portland -> 0

so no partition is empty and one of them carries two stations. Both halves of the
rule are therefore watchable: a key always lands in the same partition, and a
partition may hold more than one key. Ordering is guaranteed per partition, not
per key.

    --seconds N   stop after N seconds instead of running until Ctrl-C
    --rate N      readings per second (default 1)
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone

from kafka import KafkaProducer

# Four stations, each wandering around its own baseline temperature. Fixed seed,
# so two runs of the same length publish the same readings. See the note above on
# why these four names in particular.
STATIONS = {
    "boston": 12.0,
    "denver": 4.0,
    "chicago": 8.0,
    "portland": 11.0,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", default="localhost:9092")
    ap.add_argument("--topic", default="weatherdata")
    ap.add_argument("--rate", type=float, default=1.0, help="readings per second")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 means run until Ctrl-C")
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        key_serializer=lambda k: k.encode("utf-8"),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"publishing to {args.topic} at {args.bootstrap}, {args.rate}/s", flush=True)
    started = time.monotonic()
    sent = 0
    try:
        while True:
            if args.seconds and time.monotonic() - started >= args.seconds:
                break
            station = rng.choice(list(STATIONS))
            reading = {
                "station": station,
                # The event time: when the sensor took the measurement. The
                # streaming query windows on this column, not on its own clock.
                "reading_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "temperature_c": round(STATIONS[station] + rng.gauss(0, 1.5), 2),
            }
            producer.send(args.topic, key=station, value=reading)
            sent += 1
            if sent % 10 == 0:
                print(f"sent {sent}", flush=True)
            time.sleep(1.0 / args.rate)
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        producer.flush()
        producer.close()

    print(f"SENT={sent}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
