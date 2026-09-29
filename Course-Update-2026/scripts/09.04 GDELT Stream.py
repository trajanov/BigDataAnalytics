"""CS-777 Chapter 9 — the GDELT streaming pipeline.

A directory is the stream. Each GDELT export that appears in it is one arrival of
new data, and `09.03 GDELT Feed.py` is what puts them there, from the live GDELT
endpoint. Run the two together, each in its own terminal:

    .venv/bin/python3 "code/scripts/09.03 GDELT Feed.py"
    .venv/bin/python3 "code/scripts/09.04 GDELT Stream.py"

What the pipeline does, in order: declare the 61-field GDELT schema, read the
watched directory as a stream, join each event against the static CAMEO country
table, average the Goldstein scale per country, and publish three rankings as
three independent streaming queries.

Two details carry most of the chapter's argument.

*   `spark.read` and `spark.readStream` appear in the same script. The country
    table is bounded and read once; the events are unbounded and never finish.
    The join between them is written exactly like any other join, because a
    stream-static join needs no streaming state at all.

*   **There is no checkpoint.** This script keeps no record of which files it
    has already processed, so stopping it and starting it again reprocesses the
    whole directory. That is the behaviour Exercise 5 asks you to observe,
    explain and repair.

After every micro-batch the per-country table is also written to
./gdelt-dashboard/latest.json, which `09.05 GDELT Dashboard.py` displays.
Stop the program with Ctrl-C.
"""

import json
import os
import sys
import tempfile
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, desc
from pyspark.sql.types import StringType, StructField, StructType

HERE = os.path.dirname(os.path.abspath(__file__))
# Relative to wherever the program is started, so the files are easy to find.
INPUT_PATH = os.path.join(os.getcwd(), "gdelt-input")
DASHBOARD_PATH = os.path.join(os.getcwd(), "gdelt-dashboard")
# The data folder sits beside the scripts folder.
COUNTRIES_PATH = os.path.join(HERE, "..", "data", "gdelt", "CAMEO.country.txt")
TOP = 10  # countries in each console ranking

# The GDELT 2.0 export format: 61 tab-separated fields, every one read as a
# string and cast later where it is needed. A file-source stream cannot infer a
# schema, so it has to be declared before the first file arrives.
schema = StructType([
    StructField("GLOBALEVENTID", StringType(), True),
    StructField("SQLDATE", StringType(), True),
    StructField("MonthYear", StringType(), True),
    StructField("Year", StringType(), True),
    StructField("FractionDate", StringType(), True),
    StructField("Actor1Code", StringType(), True),
    StructField("Actor1Name", StringType(), True),
    StructField("Actor1CountryCode", StringType(), True),
    StructField("Actor1KnownGroupCode", StringType(), True),
    StructField("Actor1EthnicCode", StringType(), True),
    StructField("Actor1Religion1Code", StringType(), True),
    StructField("Actor1Religion2Code", StringType(), True),
    StructField("Actor1Type1Code", StringType(), True),
    StructField("Actor1Type2Code", StringType(), True),
    StructField("Actor1Type3Code", StringType(), True),
    StructField("Actor2Code", StringType(), True),
    StructField("Actor2Name", StringType(), True),
    StructField("Actor2CountryCode", StringType(), True),
    StructField("Actor2KnownGroupCode", StringType(), True),
    StructField("Actor2EthnicCode", StringType(), True),
    StructField("Actor2Religion1Code", StringType(), True),
    StructField("Actor2Religion2Code", StringType(), True),
    StructField("Actor2Type1Code", StringType(), True),
    StructField("Actor2Type2Code", StringType(), True),
    StructField("Actor2Type3Code", StringType(), True),
    StructField("IsRootEvent", StringType(), True),
    StructField("EventCode", StringType(), True),
    StructField("EventBaseCode", StringType(), True),
    StructField("EventRootCode", StringType(), True),
    StructField("QuadClass", StringType(), True),
    StructField("GoldsteinScale", StringType(), True),
    StructField("NumMentions", StringType(), True),
    StructField("NumSources", StringType(), True),
    StructField("NumArticles", StringType(), True),
    StructField("AvgTone", StringType(), True),
    StructField("Actor1Geo_Type", StringType(), True),
    StructField("Actor1Geo_FullName", StringType(), True),
    StructField("Actor1Geo_CountryCode", StringType(), True),
    StructField("Actor1Geo_ADM1Code", StringType(), True),
    StructField("Actor1Geo_ADM2Code", StringType(), True),
    StructField("Actor1Geo_Lat", StringType(), True),
    StructField("Actor1Geo_Long", StringType(), True),
    StructField("Actor1Geo_FeatureID", StringType(), True),
    StructField("Actor2Geo_Type", StringType(), True),
    StructField("Actor2Geo_FullName", StringType(), True),
    StructField("Actor2Geo_CountryCode", StringType(), True),
    StructField("Actor2Geo_ADM1Code", StringType(), True),
    StructField("Actor2Geo_ADM2Code", StringType(), True),
    StructField("Actor2Geo_Lat", StringType(), True),
    StructField("Actor2Geo_Long", StringType(), True),
    StructField("Actor2Geo_FeatureID", StringType(), True),
    StructField("ActionGeo_Type", StringType(), True),
    StructField("ActionGeo_FullName", StringType(), True),
    StructField("ActionGeo_CountryCode", StringType(), True),
    StructField("ActionGeo_ADM1Code", StringType(), True),
    StructField("ActionGeo_ADM2Code", StringType(), True),
    StructField("ActionGeo_Lat", StringType(), True),
    StructField("ActionGeo_Long", StringType(), True),
    StructField("ActionGeo_FeatureID", StringType(), True),
    StructField("DATEADDED", StringType(), True),
    StructField("SOURCEURL", StringType(), True)
])

# The country lookup: CAMEO code to country name, two columns with a header row.
country_schema = StructType([
    StructField("CountryCode", StringType(), True),
    StructField("CountryName", StringType(), True)
])


def main() -> int:
    os.makedirs(INPUT_PATH, exist_ok=True)
    os.makedirs(DASHBOARD_PATH, exist_ok=True)
    scratch = os.path.join(tempfile.gettempdir(), "cs777", "gdelt-stream")
    os.makedirs(scratch, exist_ok=True)

    spark = (
        SparkSession.builder.appName("CS777-9.4 GDELT CSV Streaming Processing")
        .master("local[*]")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.shuffle.partitions", 8)
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.warehouse.dir", os.path.join(scratch, "warehouse"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Read streaming data from CSV files. One file per micro-batch, so each
    # arrival is visible as its own numbered batch on the console.
    df = (
        spark.readStream.option("header", "false")
        .option("delimiter", "\t")
        .option("maxFilesPerTrigger", 1)
        .schema(schema)
        .csv(INPUT_PATH)
    )

    # Convert GoldsteinScale to float
    df = df.withColumn("GoldsteinScale", col("GoldsteinScale").cast("float"))

    # Load country code mapping file: a bounded read, in the same script
    country_mapping_df = (
        spark.read.option("header", "true")
        .option("delimiter", "\t")
        .schema(country_schema)
        .csv(COUNTRIES_PATH)
    )

    # Join the GDELT data with the country mapping to get country names
    joined_df = df.join(
        country_mapping_df,
        df.Actor1CountryCode == country_mapping_df.CountryCode,
        "left",
    )

    # Group by country name and calculate the average of GoldsteinScale and count of events
    country_goldstein_df = joined_df.groupBy("CountryName").agg(
        avg("GoldsteinScale").alias("AverageGoldsteinScale"),
        count("GLOBALEVENTID").alias("NumberOfEvents"),
    )

    rankings = {
        "positive": country_goldstein_df.orderBy(desc("AverageGoldsteinScale")).limit(TOP),
        "negative": country_goldstein_df.orderBy("AverageGoldsteinScale").limit(TOP),
        "events": country_goldstein_df.orderBy(desc("NumberOfEvents")).limit(TOP),
    }

    # Each writeStream starts an independent query: its own progress and its
    # own read of the source.
    queries = []
    for name, ranking in rankings.items():
        writer = (
            ranking.writeStream.queryName(f"gdelt-{name}")
            .outputMode("complete")
            .format("console")
            .option("numRows", TOP)
            .option("truncate", "false")
        )
        queries.append(writer.start())

    # The dashboard is one more sink. foreachBatch hands over each micro-batch's
    # result as an ordinary DataFrame: one row per country, small enough to
    # collect, so it is written as a single JSON file that 09.05 reads. The file
    # is written beside its final name and moved into place, so the web app
    # never reads a half-written snapshot.
    latest = os.path.join(DASHBOARD_PATH, "latest.json")
    history = []

    def publish(batch_df, batch_id):
        countries = [row.asDict() for row in batch_df.collect()]
        total = sum(c["NumberOfEvents"] for c in countries)
        history[:] = [h for h in history if h["batch"] != batch_id]
        history.append({"batch": batch_id, "time": time.strftime("%H:%M:%S"),
                        "events": total})
        snapshot = {"batch": batch_id, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "countries": countries, "history": history}
        with open(latest + ".tmp", "w") as handle:
            json.dump(snapshot, handle)
        os.replace(latest + ".tmp", latest)
        print(f"dashboard: batch {batch_id} written to {latest}", flush=True)

    queries.append(
        country_goldstein_df.writeStream.queryName("gdelt-dashboard")
        .outputMode("complete")
        .foreachBatch(publish)
        .start()
    )

    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
