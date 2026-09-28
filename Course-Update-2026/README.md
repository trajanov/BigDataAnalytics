# Course-Update-2026

Code library for the 2026 revision of MET CS 777. Notebooks and scripts are numbered
`CC.NN Title` — `CC` is the lecture-notes chapter, `NN` the position within it — so
`03.04 Parquet vs CSV.ipynb` is the fourth code artifact of Chapter 3.

Built and tested against **PySpark 4.2**, Python 3.12, JDK 21.

## Folder structure

### [`notebooks/`](notebooks)
The main library. Chapter 1 covers when a cluster is justified, word count, and PySpark
versus plain Python; Chapter 2 covers RDDs — Spark Connect, laziness, by-key aggregators,
partitioning, jobs/stages/tasks, and joins; Chapter 3 covers DataFrames — the DataFrame
API, window functions, Parquet versus CSV, Catalyst/AQE and skew, the `VARIANT` type, and
the taxi dataset; Chapter 4 covers data transformation — reader policy, cleaning, dates and
strings, joins, array functions, UDF cost, and validation; Chapter 5 covers performance —
the Spark UI, configuration, partitions and spill, caching, broadcast variables and
accumulators. Chapter 6 covers vectors and distributed matrices; Chapter 7 covers Spark
MLlib — pipelines, custom transformers, evaluators, grid search, TF-IDF, and clustering;
Chapter 8 implements distributed ML algorithms — linear and logistic regression by
gradient descent, regularization, mini-batches, and numerical stability; Chapter 9 covers
Structured Streaming — event-time windows, checkpoint recovery, output modes, idempotent
`foreachBatch`, and Kafka; Chapter 10 covers right-sizing, approximate aggregation,
sampling, and visualization at scale; Chapter 11 covers storage — object stores, file
formats, partitioning and bucketing, Delta Lake, and Iceberg. `images/` holds the SVG
figures the notebooks embed.

### [`cloud/`](cloud)
Scripts meant to be submitted to a serverless Spark service rather than run in a notebook.
See the folder's own README for the submission commands.

### [`data/`](data)
Only the small inputs are versioned here:

| File | Size | Used by |
|------|------|---------|
| `advertising.csv` | 4 KB | regression examples |
| `bike-sharing.csv` | 1.1 MB | 07.01, 07.05 |
| `netflix-subscription.csv` | 4 KB | 07.03, 07.09, 10.04 |
| `Alices-Adventures-in-Wonderland-by-Lewis-Carroll.txt.bz2` | 52 KB | word count |
| `taxi-data-sorted-verysmall.csv` | 1.8 MB | taxi examples |
| `taxi-data-sorted-verysmall-header.csv` | 1.8 MB | taxi examples |

The large datasets — the GDELT event tables (`data/gdelt/`, up to 4.7 GB),
`taxi-data-sorted-small.csv.bz2` (93 MB), and `IMDB Dataset.csv.bz2` (18 MB, used by
07.06–07.08) — are **not** in this repository; they are too large
to version here. Notebooks that reference them expect the files under a local
`data/` directory alongside these, or a cloud storage path. Build the GDELT tables with the
collection scripts, or download the taxi data from the course materials.
