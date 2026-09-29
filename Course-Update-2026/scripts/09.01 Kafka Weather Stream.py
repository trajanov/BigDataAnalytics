"""CS-777 Chapter 9 — the Spark half of the Kafka weather example.

Reads the `weatherdata` topic, parses the JSON payload against a declared schema,
and keeps a windowed average temperature per station. It is the complete path the
chapter describes: producer, topic, Structured Streaming consumer, sink.

Run the broker and the producer first, each in its own terminal:

    code/scripts/kafka/start.sh
    .venv/bin/python3 "code/scripts/09.01 Weather Producer.py"
    .venv/bin/python3 "code/scripts/09.01 Kafka Weather Stream.py"

Three things in this file are worth reading closely.

1.  Kafka hands Spark bytes. `key` and `value` arrive as `binary` because Kafka
    stores records as uninterpreted byte arrays, so the first thing every Kafka
    query does is cast them and parse them against a schema it declares itself.

2.  The Kafka connector ships separately from Spark. It is pulled in here through
    `spark.jars.packages` rather than `spark-submit --packages` so that the script
    also runs under a plain `python3`; either way the artifact is the same, and
    without it the query fails at planning with `Failed to find data source: kafka`.

3.  The query windows on `reading_time`, the sensor's own clock, not on Spark's.
    A reading delayed in the network still lands in the window where it belongs,
    and the watermark is what eventually lets Spark forget that window.

Do not expect this query to show up in `kafka-consumer-groups.sh --list`. Spark's
Kafka source assigns itself the partitions rather than joining a consumer group,
and it commits no offsets back to Kafka, so the group listing stays empty while
the query runs. Progress lives in Spark's checkpoint instead, which is what makes
the recovery story Spark's own rather than Kafka's.

    --seconds N   stop after N seconds (default 90); 0 runs until Ctrl-C
    --trigger T   micro-batch interval (default "5 seconds"); "" runs flat out
    --checkpoint  keep progress in a checkpoint directory, so a restart resumes
"""

import argparse
import os
import sys
import tempfile

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, from_json, window
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0"

# The payload schema, declared rather than inferred. A streaming query cannot
# infer a schema from data it has not seen yet.
READING = StructType(
    [
        StructField("station", StringType(), True),
        StructField("reading_time", TimestampType(), True),
        StructField("temperature_c", DoubleType(), True),
    ]
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", default="localhost:9092")
    ap.add_argument("--topic", default="weatherdata")
    ap.add_argument("--window", default="30 seconds")
    ap.add_argument("--watermark", default="1 minute")
    ap.add_argument("--seconds", type=float, default=90.0, help="0 runs until Ctrl-C")
    ap.add_argument(
        "--trigger",
        default="5 seconds",
        help='micro-batch interval; empty string runs batches as fast as they finish',
    )
    ap.add_argument("--checkpoint", default="", help="checkpoint directory; empty means none")
    ap.add_argument(
        "--starting-offsets",
        default="latest",
        choices=["latest", "earliest"],
        help="where in the topic to begin when there is no checkpoint",
    )
    args = ap.parse_args()

    scratch = os.path.join(tempfile.gettempdir(), "cs777", "kafka-weather")
    os.makedirs(scratch, exist_ok=True)

    spark = (
        SparkSession.builder.appName("CS777-9.1 Kafka Weather")
        .master("local[*]")
        .config("spark.jars.packages", KAFKA_PACKAGE)
        .config("spark.sql.shuffle.partitions", 8)
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.warehouse.dir", os.path.join(scratch, "warehouse"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Every row read from Kafka carries key, value, topic, partition, offset,
    # timestamp and timestampType. The payload is the binary pair; the rest is
    # metadata describing where the record came from.
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap)
        .option("subscribe", args.topic)
        .option("startingOffsets", args.starting_offsets)
        .load()
    )

    readings = (
        raw.selectExpr("CAST(key AS STRING) AS station_key", "CAST(value AS STRING) AS payload")
        .select("station_key", from_json(col("payload"), READING).alias("r"))
        .select("station_key", "r.*")
    )

    windowed = (
        readings.withWatermark("reading_time", args.watermark)
        .groupBy(window(col("reading_time"), args.window), col("station"))
        .agg(
            avg("temperature_c").alias("avg_temperature_c"),
            count("*").alias("readings"),
        )
    )

    writer = (
        windowed.writeStream.outputMode("update")
        .format("console")
        .option("truncate", "false")
        .option("numRows", 20)
    )
    # Without an interval Spark starts the next micro-batch the moment the last
    # one finishes, which on a stream of a few records per second means a batch
    # per record and a console nobody can read. The interval does not change the
    # result, only how often it is published.
    if args.trigger:
        writer = writer.trigger(processingTime=args.trigger)
    if args.checkpoint:
        writer = writer.option("checkpointLocation", args.checkpoint)

    query = writer.start()
    print(f"query started: {query.id}", flush=True)

    if args.seconds:
        query.awaitTermination(args.seconds)
        last = query.lastProgress
        query.stop()
        # A short, greppable summary, so a run can be checked without reading the
        # whole console sink. A batch that read nothing carries no state operator
        # at all, which is what a checkpointed restart with no new data looks
        # like, so the state line is conditional rather than assumed.
        if last:
            print(f"BATCHES={last['batchId'] + 1}", flush=True)
            operators = last.get("stateOperators") or []
            if operators:
                print(f"STATE_ROWS={operators[0]['numRowsTotal']}", flush=True)
            else:
                print("STATE_ROWS=none (the last batch read no new records)", flush=True)
    else:
        query.awaitTermination()

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
