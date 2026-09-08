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
the taxi dataset. `images/` holds the SVG figures the notebooks embed.

### [`cloud/`](cloud)
Scripts meant to be submitted to a serverless Spark service rather than run in a notebook.
See the folder's own README for the submission commands.

### [`data/`](data)
Only the small inputs are versioned here:

| File | Size | Used by |
|------|------|---------|
| `advertising.csv` | 4 KB | regression examples |
| `Alices-Adventures-in-Wonderland-by-Lewis-Carroll.txt.bz2` | 52 KB | word count |
| `taxi-data-sorted-verysmall.csv` | 1.8 MB | taxi examples |
| `taxi-data-sorted-verysmall-header.csv` | 1.8 MB | taxi examples |

The large datasets — the GDELT event tables (`data/gdelt/`, up to 4.7 GB) and
`taxi-data-sorted-small.csv.bz2` (93 MB) — are **not** in this repository; they exceed
GitHub's file-size limits. Notebooks that reference them expect the files under a local
`data/` directory alongside these, or a cloud storage path. Build the GDELT tables with the
collection scripts, or download the taxi data from the course materials.
