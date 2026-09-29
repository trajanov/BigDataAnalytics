# `code/scripts/` — programs that are not notebooks

A notebook is the wrong vehicle for a program that never finishes. The streaming
applications of Chapter 9 are long-running services: they are started in one terminal,
fed from another, and stopped with Ctrl-C. Running them that way is part of what the
chapter teaches, so they are scripts.

Filenames follow the same `CC.NN Title.ext` convention as `code/notebooks/`, so they
carry spaces. **Quote every path**; the commands below already do.

Every command assumes the repository root as the working directory and

```bash
source spark-env.sh
```

beforehand, which sets `JAVA_HOME`, points PySpark at `.venv`, and keeps Spark's scratch
directories off Google Drive.

---

## Chapter 5

### `05.02 Effective Config.py`

Prints the configuration a session actually ended up with, after defaults, the config
files and the command line have all had their say. See the chapter 5 section of
`code_index.md`.

---

## Chapter 9 — streaming

Seven programs in three groups. Each group is a complete example and can be run on its
own.

| Group  | Files                                                                                                   | Needs                                  |
| ------ | ------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| Kafka  | `09.01 Weather Producer.py`, `09.01 Kafka Weather Stream.py`, `kafka/start.sh`, `kafka/stop.sh` | Docker, and the Kafka connector        |
| Socket | `09.02 Socket Word Count.py`                                                                          | `nc`                                 |
| GDELT  | `09.03`–`09.05` GDELT Feed, Stream and Dashboard, `09.06 Run GDELT Demo.sh`                          | network access to data.gdeltproject.org; `flask` for 09.05 |

### 9.1 — Kafka, from producer to windowed average

A single-broker Kafka runs in Docker. It is in **KRaft mode**, so there is no ZooKeeper
container: one server carries both the broker and the controller role, which is the
combined mode the chapter describes for development.

```bash
code/scripts/kafka/start.sh
```

That is the whole of it. The script finds Docker even when it is not on your `PATH`,
brings the cluster up, waits until the broker actually answers rather than trusting the
exit status, and prints the next two commands. Running it again when the cluster is
already up is harmless.

The first start pulls about 700 MB of images and takes a minute or two. When it settles,
**Kafka UI is at http://localhost:8080** and is worth leaving open: the topic, its three
partitions, the offsets and the messages themselves are all visible there.

One thing you will *not* find there is the streaming query. Spark's Kafka source assigns
itself partitions directly instead of joining a consumer group, and it commits no offsets
back to Kafka, so `kafka-consumer-groups.sh --list` stays empty for the whole run. That is
not a misconfiguration; it is the chapter's point that Spark keeps its progress in its own
checkpoint rather than in Kafka.

Then, in two more terminals:

```bash
.venv/bin/python3 "code/scripts/09.01 Weather Producer.py"
.venv/bin/python3 "code/scripts/09.01 Kafka Weather Stream.py"
```

The producer publishes one JSON reading per second from four weather stations, keyed by
station name. The consumer reads the topic, parses the payload against a declared schema,
and prints a windowed average temperature per station, revised every five seconds.

The four station names are chosen so that they fall across all three partitions, with
`boston` and `denver` sharing one. Kafka UI shows the result, and so does

```bash
docker exec cs777-kafka /opt/kafka/bin/kafka-get-offsets.sh \
    --bootstrap-server localhost:9092 --topic weatherdata
```

which is worth running once: a key always lands in the same partition, a partition may
carry several keys, and ordering is promised per partition rather than per key.

The Spark side needs the `spark-sql-kafka-0-10` connector, which ships separately from
Spark. The script pulls it in through `spark.jars.packages`, so it works under a plain
`python3`; the first run downloads it from Maven Central and later runs use the cache.
Submitting it instead is equivalent:

```bash
.venv/bin/spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0 \
    "code/scripts/09.01 Kafka Weather Stream.py"
```

Without the connector the query fails during planning with `Failed to find data source: kafka`, which looks like an unsupported source and is really a missing jar.

Useful arguments: `--rate` and `--seconds` on the producer, `--window`, `--watermark`,
`--trigger`, `--starting-offsets earliest` and `--checkpoint DIR` on the consumer.

Stop the cluster with

```bash
code/scripts/kafka/stop.sh
```

**Topics survive an ordinary stop.** The broker's log lives in a named volume, so the next
`start.sh` finds the cluster as you left it, and the script says so by listing the topics
it found. That is worth seeing at least once: it is the retention the chapter describes,
and it is why a consumer can replay a topic from the beginning rather than only reading
what arrives next.

When you want a clean slate, either of these gives one:

```bash
code/scripts/kafka/stop.sh --purge     # stop and discard the data
code/scripts/kafka/start.sh --fresh    # discard the data, then start
```

Both scripts take `--help`, and neither touches anything on the machine beyond these two
containers.

### 9.2 — The socket word count

Two terminals. In the first, a server that forwards whatever you type:

```bash
nc -lk 9999
```

On Windows this is usually run inside the Windows Subsystem for Linux. In the second:

```bash
.venv/bin/spark-submit "code/scripts/09.02 Socket Word Count.py"
```

Type words into the Netcat window and the counts appear, revised after each micro-batch.
Update mode means each batch prints only the words whose counts changed.

The socket source is a demonstration source and nothing more. It has no replayable
offsets, so a failed query cannot recover, and its listening server is on the driver.

### 9.3 and 9.4 — the GDELT pipeline

GDELT publishes a worldwide event export every fifteen minutes. The feeder puts those
exports into a directory; the streaming query treats that directory as its input.

```bash
.venv/bin/python3 "code/scripts/09.03 GDELT Feed.py"
.venv/bin/python3 "code/scripts/09.04 GDELT Stream.py"
```

The feeder writes to `gdelt-input/` in the folder you run it from, creating it on demand.
Start both programs from the same folder, so the stream watches the directory the feeder
writes to. The country lookup comes from `code/data/gdelt/CAMEO.country.txt`.

Each export arrives as its own micro-batch, and the console shows three rankings updating
together: the most positive countries by average Goldstein scale, the most negative, and
the busiest by event count.

**On the checkpoint, and Exercise 5.** `09.04 GDELT Stream.py` keeps no checkpoint. Stop
it and start it again and it will reprocess every file in the watched directory, because
it has no record of what it has already seen. That is the behaviour Exercise 5 asks you
to observe, explain and repair.

### 9.5 — a live dashboard over the GDELT stream

Spark does not serve web pages; a streaming query writes to a sink, and another program
reads the sink. `09.04` runs a fourth query whose `foreachBatch` writes the full
per-country table to `gdelt-dashboard/latest.json` after every micro-batch.
`09.05 GDELT Dashboard.py` is a small Flask app that serves a page with
four charts (most positive, most negative, busiest countries, and events per batch)
and refreshes them every five seconds. Run the three programs from the same folder:

```bash
.venv/bin/python3 "code/scripts/09.03 GDELT Feed.py"
.venv/bin/python3 "code/scripts/09.04 GDELT Stream.py"
.venv/bin/python3 "code/scripts/09.05 GDELT Dashboard.py"
```

and open <http://127.0.0.1:8050>. The page loads Chart.js from a CDN. Countries with
fewer than five events are left out of the Goldstein rankings, so a single event cannot
top the chart.

On macOS, `09.06 Run GDELT Demo.sh` starts all three in their own Terminal windows from
the current folder and opens the dashboard in the browser:

```bash
"code/scripts/09.06 Run GDELT Demo.sh"
```
