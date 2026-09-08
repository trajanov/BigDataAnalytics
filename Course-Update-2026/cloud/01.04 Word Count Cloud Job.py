#!/usr/bin/env python3
"""CS-777 1.4 -- Word count as a serverless Spark job.

Chapter 1, Exercise 8: run word count locally, then run the *same* pipeline against a
much larger file in object storage, submitted as a serverless job, and account for the
difference in wall-clock time.

The pipeline here is deliberately identical to the one in
`code/notebooks/01.02 Word Count.ipynb`.  Reusing it is the control in the experiment:
the only things that differ between the two runs are the size of the input and the
place the machines live.

Arguments are positional, in the order the cloud appendix uses:

    <input>   a path or URI readable by the executors:
              a local path, gs://bucket/key, s3a://bucket/key, hdfs://...
    <output>  a directory (NOT a file) that does not already exist

Local:

    source spark-env.sh
    .venv/bin/spark-submit "code/cloud/01.04 Word Count Cloud Job.py" \
        code/data/Alices-Adventures-in-Wonderland-by-Lewis-Carroll.txt.bz2 \
        /tmp/cs777/wordcount-out

Serverless (Google Cloud Managed Service for Apache Spark).  See the cloud appendix for
account setup, buckets and permissions; the script must be uploaded to a bucket first,
because the service reads it from object storage rather than from your machine:

    gcloud dataproc batches submit pyspark \
        "gs://YOUR-BUCKET/01.04 Word Count Cloud Job.py" \
        --region=us-east1 --version=2.2 \
        -- gs://YOUR-BUCKET/corpus.txt gs://YOUR-BUCKET/wordcount-out/

Everything after the bare `--` is passed to this script as its arguments.

The job prints one machine-readable line to the driver log:

    ELAPSED_SECONDS=<n>

That line is the deliverable of Exercise 8.  It is printed rather than displayed because a
serverless job leaves no cluster and no Spark web interface behind: the driver log is the
only place a number can be recovered from afterwards.
"""

import argparse
import re
import sys
import time

from pyspark.sql import SparkSession

# The word pattern carries both apostrophe forms.  Public-domain texts from Project
# Gutenberg use U+2019, and without it "don't" splits into "don" and "t".
WORD_RE = re.compile(r"[a-z'\u2019]+")


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Word count over a text corpus, for local and serverless runs.")
    p.add_argument("input", help="input path or URI (local, gs://, s3a://, hdfs://)")
    p.add_argument("output", help="output DIRECTORY; must not already exist")
    p.add_argument("--top", type=int, default=100,
                   help="how many of the most frequent words to write (default: 100)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    # No .master() here.  Locally, spark-submit supplies it; on the cluster, the service
    # does.  Hard-coding local[*] in a script that is meant to be submitted is one of the
    # more common ways for a cloud job to quietly run on one machine.
    spark = (SparkSession.builder
             .appName("CS777-1.4-word-count")
             .getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    print(f"spark version : {spark.version}")
    print(f"master        : {sc.master}")
    print(f"input         : {args.input}")
    print(f"output        : {args.output}")

    lines = sc.textFile(args.input)

    # Time the action only.  Everything above this point is lazy and costs nothing;
    # everything below is where the work actually happens.
    t0 = time.time()

    counts = (lines
              .flatMap(lambda line: WORD_RE.findall(line.lower()))
              .map(lambda w: (w, 1))
              .reduceByKey(lambda a, b: a + b))

    # take(n) on a sorted RDD rather than top(n) into the driver and then parallelize:
    # sortBy is a wide dependency, but the result written out is already small.
    top = counts.sortBy(lambda kv: -kv[1]).take(args.top)

    elapsed = time.time() - t0

    # Write the deliverable out.  A serverless job's stdout is not a deliverable: the
    # cluster is gone by the time you read it.
    (sc.parallelize(top, 1)
       .map(lambda kv: f"{kv[1]}\t{kv[0]}")
       .saveAsTextFile(args.output))

    print(f"distinct_words={counts.count()}")
    print(f"top_written={len(top)} to {args.output}")
    print(f"ELAPSED_SECONDS={elapsed:.2f}")

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
