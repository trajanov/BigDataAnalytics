"""
CS-777, chapter 8 — the RDD gradient loop against its DataFrame equivalent.

The script form of the measurements in `code/notebooks/08.08 DataFrame Versions and
Their Speed.ipynb`, so that the same comparison can be run somewhere other than a
laptop. Nothing about the arithmetic differs from the notebook; what differs is that
the data is generated on the executors rather than shipped from the driver, which is
what lets the row counts go past what a single machine can hold.

Usage
-----
    spark-submit "08.08 DataFrame vs RDD Benchmark.py" [options]

    --sizes       comma-separated row counts       (default 25000,100000,400000)
    --iters       gradient descent iterations      (default 10)
    --repeats     timed repeats, median is kept    (default 3)
    --partitions  fixed partition count            (default: derived from the row count)
    --rows-per-partition   target partition size    (default 250000)
    --check-max   above this, the agreement check runs 2 iterations (default 2000000)
    --no-swap     skip the settings-swap section
    --df-only     time only the DataFrame loops, on data generated in the engine
    --swap-size   rows for the settings swap        (default: the largest of --sizes)
    --out         optional gs:// or local path for a CSV of the results

Every result line is prefixed `RESULT` so it can be recovered from a driver log that
contains nothing else:

    gcloud logging read 'resource.type="cloud_dataproc_batch"' --limit 500 | grep RESULT

No input data is read. The script synthesizes what it needs from a fixed seed, so a
run on a cluster and a run on a laptop are measuring the same computation.
"""

import argparse
import math
import operator
import sys
import time
from functools import reduce

import numpy as np
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, StructField, StructType

D = 3                                                  # features, before the intercept
COLS = ["one", "x1", "x2", "x3"]                       # model columns, in parameter order
THETA_TRUE = np.array([5.0, -1.0, 1.0, 2.0])
BETA_TRUE = np.array([-0.5, 1.5, -1.0, 2.0])
SEED = 777


# --------------------------------------------------------------------------- data


def make_data(spark, n_rows, parts, kind):
    """Build the same dataset twice, as an RDD of (label, vector) and as a DataFrame.

    Generation happens on the executors. Shipping several million NumPy arrays out of
    the driver, which is what the notebook does at laptop scale, becomes the dominant
    cost long before the gradient loop does.
    """
    sc = spark.sparkContext
    per = max(1, n_rows // parts)

    def gen(idx):
        rng = np.random.default_rng(SEED + idx)
        X = np.hstack([np.ones((per, 1)), rng.random((per, D))])
        if kind == "linear":
            y = X @ THETA_TRUE + rng.random(per) * 0.1
        else:
            logits = X @ BETA_TRUE
            y = (rng.random(per) < 1.0 / (1.0 + np.exp(-logits))).astype(float)
        # Yield rather than return a list: at ten million rows the list of tuples is
        # several hundred megabytes per partition and the array it is built from is not.
        for i in range(per):
            yield (float(y[i]), X[i])

    rdd = sc.parallelize(range(parts), parts).flatMap(gen).cache()
    n = rdd.count()

    schema = StructType([StructField(c, DoubleType(), False)
                         for c in ("y", "x1", "x2", "x3")])
    df = (spark.createDataFrame(
              rdd.map(lambda r: (r[0], float(r[1][1]), float(r[1][2]), float(r[1][3]))),
              schema)
             .withColumn("one", F.lit(1.0))
             .select("y", *COLS)
             .repartition(parts)
             .cache())
    df.count()
    return rdd, df, n


def make_df_native(spark, n_rows, parts, kind):
    """Build the DataFrame in the engine, with no Python anywhere in the construction.

    `make_data` above builds the DataFrame from the RDD, so that both representations hold
    byte-identical data and can be checked against each other. That construction sends every
    row through a Python worker, which is fine at ten million and is the dominant cost at a
    hundred million — and it is not the cost under study.

    This generator is for `--df-only` runs, where there is no RDD to agree with. The columns
    are the same shape and distribution; the individual values are not the same values.
    """
    df = (spark.range(0, n_rows, step=1, numPartitions=parts)
               .withColumn("one", F.lit(1.0))
               .withColumn("x1", F.rand(seed=SEED + 1))
               .withColumn("x2", F.rand(seed=SEED + 2))
               .withColumn("x3", F.rand(seed=SEED + 3)))

    lin = (F.lit(float(THETA_TRUE[0])) * F.col("one")
           + F.lit(float(THETA_TRUE[1])) * F.col("x1")
           + F.lit(float(THETA_TRUE[2])) * F.col("x2")
           + F.lit(float(THETA_TRUE[3])) * F.col("x3"))

    if kind == "linear":
        df = df.withColumn("y", lin + F.rand(seed=SEED + 4) * F.lit(0.1))
    else:
        logits = (F.lit(float(BETA_TRUE[0])) * F.col("one")
                  + F.lit(float(BETA_TRUE[1])) * F.col("x1")
                  + F.lit(float(BETA_TRUE[2])) * F.col("x2")
                  + F.lit(float(BETA_TRUE[3])) * F.col("x3"))
        p = F.lit(1.0) / (F.lit(1.0) + F.exp(-logits))
        df = df.withColumn("y", (F.rand(seed=SEED + 4) < p).cast("double"))

    df = df.select("y", *COLS).cache()
    n = df.count()
    return df, n


# ------------------------------------------------------------------- the four loops


def lin_gd_rdd(rdd, n, lr=1.0, iters=10):
    theta = np.zeros(D + 1)
    for _ in range(iters):
        g = rdd.map(lambda x: (x[1], (x[0] - np.dot(x[1], theta)))) \
               .map(lambda x: (x[0] * x[1], x[1] ** 2)) \
               .reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        theta = theta - lr * ((-1 / float(n)) * g[0])
    return theta


LIN_CELL = [np.zeros(D + 1)]


def _lin_seq(acc, r):
    resid = r[0] - np.dot(r[1], LIN_CELL[0])
    return (acc[0] + r[1] * resid, acc[1] + resid * resid)


def lin_gd_rdd_tree(rdd, n, lr=1.0, iters=10):
    LIN_CELL[0] = np.zeros(D + 1)
    for _ in range(iters):
        g, _sse = rdd.treeAggregate((np.zeros(D + 1), 0.0), _lin_seq,
                                    lambda a, b: (a[0] + b[0], a[1] + b[1]), depth=2)
        LIN_CELL[0] = LIN_CELL[0] - lr * ((-1 / float(n)) * g)
    return LIN_CELL[0]


def lin_gd_df(df, n, lr=1.0, iters=10):
    theta = np.zeros(D + 1)
    for _ in range(iters):
        pred = reduce(operator.add,
                      [F.lit(float(theta[j])) * F.col(c) for j, c in enumerate(COLS)])
        resid = F.col("y") - pred
        exprs = [F.sum(resid * F.col(c)).alias("g%d" % j) for j, c in enumerate(COLS)]
        exprs.append(F.sum(resid * resid).alias("sse"))
        row = df.agg(*exprs).first()
        grad = (-1 / float(n)) * np.array([row["g%d" % j] for j in range(len(COLS))])
        theta = theta - lr * grad
    return theta


def log_gd_rdd(rdd, n, lr=1.0, iters=10):
    beta = np.zeros(D + 1)
    for _ in range(iters):
        g = rdd.map(lambda x: (x[0], x[1], np.dot(x[1], beta))) \
               .treeAggregate(
                   (np.zeros(D + 1), 0.0),
                   lambda acc, r: (acc[0] + r[1] * (-r[0] + np.exp(r[2]) / (1 + np.exp(r[2]))),
                                   acc[1] - r[0] * r[2] + np.log(1 + np.exp(r[2]))),
                   lambda a, b: (a[0] + b[0], a[1] + b[1]))
        beta = beta - lr * ((1 / float(n)) * g[0])
    return beta


def log_gd_df(df, n, lr=1.0, iters=10):
    beta = np.zeros(D + 1)
    for _ in range(iters):
        theta = reduce(operator.add,
                       [F.lit(float(beta[j])) * F.col(c) for j, c in enumerate(COLS)])
        sigma = F.lit(1.0) / (F.lit(1.0) + F.exp(-theta))
        err = sigma - F.col("y")
        exprs = [F.sum(err * F.col(c)).alias("g%d" % j) for j, c in enumerate(COLS)]
        exprs.append(F.sum(-F.col("y") * theta + F.log(F.lit(1.0) + F.exp(theta))).alias("cost"))
        row = df.agg(*exprs).first()
        grad = (1 / float(n)) * np.array([row["g%d" % j] for j in range(len(COLS))])
        beta = beta - lr * grad
    return beta


# ------------------------------------------------------------------------- timing


def time_it(fn, repeats):
    fn()                                               # warm-up, not timed
    secs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        secs.append(time.perf_counter() - t0)
    return float(np.median(secs))


def emit(*fields):
    print("RESULT\t" + "\t".join(str(f) for f in fields))
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="25000,100000,400000")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--partitions", type=int, default=0,
                    help="fixed partition count; 0 means derive it from --rows-per-partition")
    ap.add_argument("--rows-per-partition", type=int, default=250_000,
                    help="target partition size when --partitions is not given")
    ap.add_argument("--check-max", type=int, default=2_000_000,
                    help="above this row count the agreement check runs 2 iterations, not --iters")
    ap.add_argument("--no-swap", action="store_true", help="skip the settings-swap section")
    ap.add_argument("--df-only", action="store_true",
                    help="time only the DataFrame loops, on data generated in the engine")
    ap.add_argument("--swap-size", type=int, default=0,
                    help="row count for the settings swap; 0 means the largest of --sizes")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    spark = SparkSession.builder.appName("CS777-8.08-DataFrame-vs-RDD").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    sizes = [int(s) for s in args.sizes.split(",")]

    def partitions_for(n):
        # A fixed partition count is wrong across three orders of magnitude of row count:
        # four partitions of a hundred million rows is 25 million rows each, and the
        # generator would build a 800 MB array per task. Size by rows instead, with at
        # least one partition per core so nothing sits idle.
        if args.partitions:
            return args.partitions
        return max(sc.defaultParallelism,
                   int(math.ceil(n / float(args.rows_per_partition))))

    emit("env", "spark_version", spark.version)
    emit("env", "default_parallelism", sc.defaultParallelism)
    emit("env", "rows_per_partition_target", args.rows_per_partition)
    for _n in sizes:
        emit("env", "partitions_for_%d" % _n, partitions_for(_n))
    emit("env", "iterations", args.iters)
    emit("env", "repeats", args.repeats)
    for k in ("spark.executor.instances", "spark.executor.cores", "spark.executor.memory",
              "spark.sql.shuffle.partitions", "spark.sql.adaptive.enabled",
              "spark.dynamicAllocation.enabled"):
        # A typed conf rejects a string default, so ask for it and catch the miss.
        try:
            emit("env", k, spark.conf.get(k))
        except Exception:
            emit("env", k, "<unset>")

    rows = []

    if args.df_only:
        emit("mode", "df_only", "data generated in the engine; no RDD built, no agreement check")
        for kind, df_fn in (("linear", lin_gd_df), ("logistic", log_gd_df)):
            for n_rows in sizes:
                parts = partitions_for(n_rows)
                df, n = make_df_native(spark, n_rows, parts, kind)

                # Not a correctness check against an RDD - there is none here. This confirms
                # the generated columns are the shape the loops assume before anything is timed.
                st = df.selectExpr("avg(x1) a1", "avg(x2) a2", "avg(x3) a3",
                                   "avg(y) ay", "min(one) mo", "max(one) xo").first()
                emit("stats", kind, n, parts,
                     round(st["a1"], 4), round(st["a2"], 4), round(st["a3"], 4),
                     round(st["ay"], 4), st["mo"], st["xo"])
                assert abs(st["mo"] - 1.0) < 1e-12 and abs(st["xo"] - 1.0) < 1e-12

                t_df = time_it(lambda: df_fn(df, n, iters=args.iters), args.repeats)
                emit("bench_df", kind, n, parts, round(t_df, 4),
                     round(t_df / args.iters, 5))
                rows.append((kind, n, float("nan"), t_df))
                df.unpersist(blocking=True)
        _finish(spark, args, rows)
        return

    for kind, rdd_fn, df_fn in (("linear", lin_gd_rdd, lin_gd_df),
                                ("logistic", log_gd_rdd, log_gd_df)):
        for n_rows in sizes:
            parts = partitions_for(n_rows)
            rdd, df, n = make_data(spark, n_rows, parts, kind)

            # Correctness first: the two representations must agree before either is timed.
            # Above --check-max a full-length check would cost as much as the benchmark, and
            # two iterations establish agreement just as well as ten.
            chk = args.iters if n <= args.check_max else 2
            a, b = rdd_fn(rdd, n, iters=chk), df_fn(df, n, iters=chk)
            agree = bool(np.allclose(a, b))
            emit("agree", kind, n, agree, chk, np.round(a, 8).tolist())
            if not agree:
                raise SystemExit("RDD and DataFrame disagree for %s at n=%d" % (kind, n))

            t_rdd = time_it(lambda: rdd_fn(rdd, n, iters=args.iters), args.repeats)
            t_df = time_it(lambda: df_fn(df, n, iters=args.iters), args.repeats)
            emit("bench", kind, n, round(t_rdd, 4), round(t_df, 4), round(t_rdd / t_df, 3))
            rows.append((kind, n, t_rdd, t_df))

            rdd.unpersist()
            df.unpersist()

    # ---- the settings swap, at the largest size, on the linear model ----------
    if args.no_swap:
        emit("swap", "skipped", 0, 0.0)
        _finish(spark, args, rows)
        return

    n_rows = args.swap_size or sizes[-1]
    rdd, df, n = make_data(spark, n_rows, partitions_for(n_rows), "linear")
    emit("swap_size", n)

    for label, fn in (("rdd_reduce", lin_gd_rdd), ("rdd_treeAggregate", lin_gd_rdd_tree)):
        emit("swap", label, n, round(time_it(lambda: fn(rdd, n, iters=args.iters), args.repeats), 4))

    for aqe in ("true", "false"):
        spark.conf.set("spark.sql.adaptive.enabled", aqe)
        lin_gd_df(df, n, iters=args.iters)
        emit("swap", "df_aqe_" + aqe, n,
             round(time_it(lambda: lin_gd_df(df, n, iters=args.iters), args.repeats), 4))
    spark.conf.set("spark.sql.adaptive.enabled", "true")

    for codegen in ("true", "false"):
        spark.conf.set("spark.sql.codegen.wholeStage", codegen)
        lin_gd_df(df, n, iters=args.iters)
        emit("swap", "df_codegen_" + codegen, n,
             round(time_it(lambda: lin_gd_df(df, n, iters=args.iters), args.repeats), 4))
    spark.conf.set("spark.sql.codegen.wholeStage", "true")

    rdd.unpersist()
    df.unpersist()

    _finish(spark, args, rows)


def _finish(spark, args, rows):
    if args.out:
        out = spark.createDataFrame(
            [(k, int(n), float(tr), float(td),
              float(tr / td) if tr == tr else float("nan")) for k, n, tr, td in rows],
            ["model", "rows", "rdd_seconds", "df_seconds", "rdd_over_df"])
        out.coalesce(1).write.mode("overwrite").option("header", True).csv(args.out)
        emit("wrote", args.out)

    emit("done")
    spark.stop()


if __name__ == "__main__":
    main()
