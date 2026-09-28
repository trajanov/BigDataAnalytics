"""
CS-777, chapter 6 — what the container a vector is stored in costs.

The script form of `code/notebooks/06.06 Vector Storage Compared.ipynb`, so that the
same comparison can be made somewhere other than a laptop. The notebook's claim is that
the ranking of containers — Python list, NumPy array, SciPy sparse matrix,
`pyspark.ml` vector, `pyspark.mllib` vector, DataFrame column — is the same on the
driver and inside a task. A laptop cannot check the second half of that claim, because
on a laptop the driver and the task are the same machine. This script can: it runs the
same three microbenchmarks in three places — the driver, inside a Python worker on every
executor, and across a whole distributed gradient — and prints all three. A fourth
section takes the vector out of Python altogether, computing the same dot product and
the same gradient with the vector SQL expressions Spark 4.2 added, where they exist.

Usage
-----
    spark-submit "06.06 Vector Storage Benchmark.py" [options]

    --dim          width of the dense micro-benchmark vector   (default 10000)
    --sparse-dim   width of the sparse vector                  (default 100000)
    --nnz          non-zeros in the sparse vector              (default 1000)
    --rows         rows in the distributed gradient            (default 100000)
    --features     features per row                            (default 200)
    --partitions   partitions for the gradient; 0 = derived    (default 0)
    --rows-per-partition  target partition size                (default 25000)
    --repeats      timed repeats; the best is kept             (default 5)
    --gradient-repeats    repeats of each distributed gradient (default 3)
    --skip-list    drop the Python-list gradient variant       (it is the slow one)
    --skip-engine  skip the engine-side section (needs Spark 4.2)
    --skip-executor-micro skip the per-executor microbenchmark
    --out          optional gs:// or local path for a CSV of the results

Every result line is prefixed `RESULT` and is tab-separated, so it can be recovered
from a driver log that contains nothing else:

    gcloud logging read 'resource.type="cloud_dataproc_batch"' --limit 3000 | grep RESULT

No input data is read. Every vector is synthesized from a fixed seed, and the rows of
the distributed section are generated **inside the tasks** from a per-partition seed, so
a run on a cluster and a run on a laptop measure the same computation and nothing large
crosses from the driver.
"""

import argparse
import math
import os
import pickle
import platform
import socket
import sys
import time
import timeit

import numpy as np
import scipy
import scipy.sparse as sps

from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors as MLVectors
from pyspark.mllib.linalg import Vectors as MLlibVectors
from pyspark.sql import SparkSession, functions as F

SEED = 777


# --------------------------------------------------------------------------- output


def emit(*fields):
    """One tab-separated result line, flushed, so a truncated log still has the rows."""
    print("RESULT\t" + "\t".join(str(f) for f in fields))
    sys.stdout.flush()


ROWS = []          # (section, container, metric, value) — the CSV written by --out


def record(section, container, metric, value):
    ROWS.append((section, container, metric, float(value)))


# ------------------------------------------------------------------------ the bench


def bench(fn, number=200, repeat=5):
    """Microseconds per call, best of `repeat` batches of `number` calls.

    Best-of is the right statistic for "how fast can this operation run": everything
    that makes a repeat slower — a garbage collection, a noisy neighbour on a shared
    cloud machine — is noise, not the operation. On a cluster the noisy neighbour is
    not hypothetical, which is why the median is not used here.
    """
    return min(timeit.repeat(fn, number=number, repeat=repeat)) / number * 1e6


def nbytes(v):
    """What the container occupies, counted the way the container is built."""
    if isinstance(v, list):
        return sys.getsizeof(v) + sum(sys.getsizeof(e) for e in v)
    if sps.issparse(v):
        return v.data.nbytes + v.indices.nbytes + v.indptr.nbytes
    if hasattr(v, "array"):                      # ml / mllib DenseVector
        return v.array.nbytes
    if hasattr(v, "values"):                     # ml / mllib SparseVector
        return v.values.nbytes + v.indices.nbytes
    return v.nbytes                              # ndarray


# ------------------------------------------------- the micro-suite, runnable anywhere


def micro_suite(dim, sparse_dim, nnz, repeats):
    """Footprint, dot, axpy, sparse dot and pickle cost for every container.

    Written as one self-contained function with no Spark in it, because it is called in
    two places: on the driver, and inside a Python worker on every executor. That is the
    whole point of the script — the same instrument, read in both places.

    Returns a list of (section, container, metric, value) tuples.
    """
    out = []

    def add(section, container, metric, value):
        out.append((section, container, metric, float(value)))

    rng = np.random.default_rng(SEED)
    w_np, x_np = rng.standard_normal(dim), rng.standard_normal(dim)
    w_list, x_list = w_np.tolist(), x_np.tolist()
    w_csr = sps.csr_matrix(w_np.reshape(1, -1))
    x_csr = sps.csr_matrix(x_np.reshape(1, -1))
    w_ml, x_ml = MLVectors.dense(w_np), MLVectors.dense(x_np)
    w_mllib, x_mllib = MLlibVectors.dense(w_np), MLlibVectors.dense(x_np)

    containers = {
        "python_list": w_list,
        "numpy_ndarray": w_np,
        "scipy_csr": w_csr,
        "ml_dense": w_ml,
        "mllib_dense": w_mllib,
    }

    # -- the assertion first: five types, one set of numbers -------------------
    same = {
        "python_list": np.array(w_list),
        "numpy_ndarray": w_np,
        "scipy_csr": w_csr.toarray().ravel(),
        "ml_dense": w_ml.toArray(),
        "mllib_dense": w_mllib.toArray(),
    }
    for name, arr in same.items():
        assert np.array_equal(arr, w_np), name

    # -- 1. footprint ----------------------------------------------------------
    for name, v in containers.items():
        b = nbytes(v)
        add("footprint", name, "bytes", b)
        add("footprint", name, "bytes_per_entry", b / dim)

    # -- 2. the dot product: one prediction ------------------------------------
    dot = {
        "python_list": lambda: sum(a * b for a, b in zip(w_list, x_list)),
        "numpy_ndarray": lambda: w_np @ x_np,
        "scipy_csr": lambda: x_csr.dot(w_np),
        "ml_dense": lambda: x_ml.dot(w_ml),
        "mllib_dense": lambda: x_mllib.dot(w_mllib),
    }
    for name, fn in dot.items():
        # The list is ~200x slower, so timing it `number` times costs `number` times as
        # much wall clock. Fewer calls of a slower operation is the same measurement.
        t = bench(fn, number=20 if name == "python_list" else 200, repeat=repeats)
        add("dot", name, "us_per_call", t)

    # -- 3. the axpy: one gradient step ----------------------------------------
    eta = 0.01
    axpy = {
        "python_list": lambda: [a - eta * b for a, b in zip(w_list, x_list)],
        "numpy_ndarray": lambda: w_np - eta * x_np,
        "ml_dense": lambda: w_ml - eta * x_ml,
        "mllib_dense": lambda: w_mllib - eta * x_mllib,
    }
    for name, fn in axpy.items():
        t = bench(fn, number=20 if name == "python_list" else 200, repeat=repeats)
        add("axpy", name, "us_per_call", t)

    # -- 4. the sparse case: d = sparse_dim with nnz non-zeros -----------------
    idx = np.sort(rng.choice(sparse_dim, nnz, replace=False))
    vals = rng.standard_normal(nnz)
    wD = rng.standard_normal(sparse_dim)
    xs_np = np.zeros(sparse_dim)
    xs_np[idx] = vals
    xs_csr = sps.csr_matrix((vals, idx, [0, nnz]), shape=(1, sparse_dim))
    xs_ml = MLVectors.sparse(sparse_dim, idx.tolist(), vals.tolist())
    xs_mllib = MLlibVectors.sparse(sparse_dim, idx.tolist(), vals.tolist())
    wD_ml, wD_mllib = MLVectors.dense(wD), MLlibVectors.dense(wD)
    idx_list, vals_list, wD_list = idx.tolist(), vals.tolist(), wD.tolist()

    sparse_dot = {
        "list_over_nonzeros": lambda: sum(wD_list[i] * v
                                          for i, v in zip(idx_list, vals_list)),
        "numpy_dense_dot_dense": lambda: xs_np @ wD,
        "scipy_csr_dot_dense": lambda: xs_csr.dot(wD),
        "ml_sparse_dot_ndarray": lambda: xs_ml.dot(wD),
        "ml_sparse_dot_dense": lambda: xs_ml.dot(wD_ml),
        "mllib_sparse_dot_dense": lambda: xs_mllib.dot(wD_mllib),
        "ml_sparse_dot_sparse": lambda: xs_ml.dot(xs_ml),
    }
    expected = xs_np @ wD
    for name, fn in sparse_dot.items():
        if name != "ml_sparse_dot_sparse":       # that one computes a different product
            assert np.isclose(fn(), expected), name
        add("sparse_dot", name, "us_per_call", bench(fn, number=50, repeat=repeats))

    dense_bytes = xs_np.nbytes
    sparse_bytes = xs_ml.indices.nbytes + xs_ml.values.nbytes
    add("sparse_footprint", "numpy_dense", "bytes", dense_bytes)
    add("sparse_footprint", "ml_sparse", "bytes", sparse_bytes)
    add("sparse_footprint", "ml_sparse", "times_smaller", dense_bytes / sparse_bytes)

    # -- 5. the cost of moving a record ----------------------------------------
    movers = {
        "python_list": w_list,
        "numpy_ndarray": w_np,
        "ml_dense": w_ml,
        "scipy_csr_dense_data": w_csr,
        "numpy_dense_sparse_data": xs_np,
        "ml_sparse": xs_ml,
    }
    for name, obj in movers.items():
        blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        add("pickle", name, "bytes", len(blob))
        add("pickle", name, "dumps_us",
            bench(lambda o=obj: pickle.dumps(o, protocol=pickle.HIGHEST_PROTOCOL),
                  number=50, repeat=repeats))
        add("pickle", name, "loads_us",
            bench(lambda b=blob: pickle.loads(b), number=50, repeat=repeats))

    return out


# ------------------------------------------------------- the distributed gradient


def build_rows(sc, n, d, parts, w_true):
    """An RDD of (ndarray, float), generated inside the tasks from a per-partition seed.

    Nothing large crosses from the driver: each task receives an index and builds its own
    rows. At a hundred thousand rows a driver-side `parallelize` of NumPy arrays would be
    measurable; at ten million it would be the whole measurement.
    """
    def make(pid, it):
        r = np.random.default_rng(pid)
        for _ in it:
            x = r.standard_normal(d)
            yield x, float(x @ w_true) + 0.1 * r.standard_normal()

    return sc.parallelize(range(n), parts).mapPartitionsWithIndex(make)


def timed(fn, repeats):
    """Best of `repeats` wall-clock runs, and the value the last run produced."""
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def materialize(x):
    t0 = time.perf_counter()
    x.cache().count()
    return time.perf_counter() - t0


def sql_float32_literal(vec):
    """A NumPy vector as a Spark SQL `ARRAY<FLOAT>` literal, e.g. for vector_inner_product.

    Used for both the gradient weight vector and the vector-search query vector, so it is
    written once rather than assembled inline at each call site. Kept only for the
    engine_residual probe below, which exists to demonstrate the slow, inline-`F.expr`
    way of writing this query; float32_array_column is the fast path.
    """
    return "array(%s)" % ",".join(
        "cast(%r as float)" % float(v) for v in np.asarray(vec, dtype=np.float32))


def float32_array_column(vec):
    """The same vector as a Spark `ARRAY<FLOAT>` Column, built through the DataFrame API
    rather than interpolated into `F.expr` SQL text.

    Two things follow from not going through text: the expression is built once and
    reused as a Column object across a query's repeats, instead of being re-tokenized
    and re-parsed by the SQL parser on every one of them; and it can hold a non-finite
    weight safely — Spark SQL has no numeric-literal syntax for NaN or Infinity, so
    %r-formatting one into `F.expr` text is a parse error, where `F.lit` passes it
    through as an ordinary float Literal.
    """
    return F.array(*[F.lit(float(v)).cast("float")
                     for v in np.asarray(vec, dtype=np.float32)])


def engine_vector_fns(spark):
    """Whether this Spark has the vector SQL expressions of part 4.

    They arrived in Spark 4.2. Dataproc Serverless runtime 3.0 ships 4.0.1, so the
    cluster run of this script reaches the negative branch — which is the reason the
    capability is probed rather than assumed from a version string.
    """
    try:
        spark.sql("SELECT vector_inner_product(array(cast(1.0 as float)),"
                  " array(cast(1.0 as float)))").first()
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=10_000)
    ap.add_argument("--sparse-dim", type=int, default=100_000)
    ap.add_argument("--nnz", type=int, default=1_000)
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--features", type=int, default=200)
    ap.add_argument("--partitions", type=int, default=0,
                    help="0 means derive it from --rows-per-partition")
    ap.add_argument("--rows-per-partition", type=int, default=25_000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--gradient-repeats", type=int, default=3)
    ap.add_argument("--skip-list", action="store_true",
                    help="drop the Python-list gradient variant")
    ap.add_argument("--skip-engine", action="store_true",
                    help="skip the engine-side section, which needs Spark 4.2 or later")
    ap.add_argument("--skip-executor-micro", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    spark = (SparkSession.builder
             .appName("CS777-6.06-Vector-Storage")
             .getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    def die(msg):
        """Stop the session before exiting, so a disagreement still tears the batch job
        down cleanly instead of leaving it for the caller's timeout."""
        spark.stop()
        raise SystemExit(msg)

    parts = args.partitions or max(
        sc.defaultParallelism,
        int(math.ceil(args.rows / float(args.rows_per_partition))))

    # ---- what machine is this ------------------------------------------------
    emit("env", "driver", "spark_version", spark.version)
    emit("env", "driver", "python_version", platform.python_version())
    emit("env", "driver", "numpy_version", np.__version__)
    emit("env", "driver", "scipy_version", scipy.__version__)
    emit("env", "driver", "platform", platform.platform())
    emit("env", "driver", "processor", platform.processor() or "unknown")
    emit("env", "driver", "cpu_count", os.cpu_count())
    emit("env", "driver", "hostname", socket.gethostname())
    emit("env", "driver", "default_parallelism", sc.defaultParallelism)
    emit("env", "driver", "master", sc.master)
    emit("env", "run", "dim", args.dim)
    emit("env", "run", "sparse_dim", args.sparse_dim)
    emit("env", "run", "nnz", args.nnz)
    emit("env", "run", "rows", args.rows)
    emit("env", "run", "features", args.features)
    emit("env", "run", "partitions", parts)
    emit("env", "run", "repeats", args.repeats)
    emit("env", "run", "gradient_repeats", args.gradient_repeats)
    for k in ("spark.executor.instances", "spark.executor.cores", "spark.executor.memory",
              "spark.dynamicAllocation.enabled", "spark.dataproc.engine"):
        try:
            emit("env", "conf", k, spark.conf.get(k))
        except Exception:
            emit("env", "conf", k, "<unset>")

    # ---- part 1: the micro-suite, on the driver ------------------------------
    t0 = time.perf_counter()
    for section, container, metric, value in micro_suite(
            args.dim, args.sparse_dim, args.nnz, args.repeats):
        emit("driver_" + section, container, metric, round(value, 4))
        record("driver_" + section, container, metric, value)
    emit("timing", "driver_micro", "seconds", round(time.perf_counter() - t0, 2))

    # ---- part 2: the same micro-suite, inside a Python worker ----------------
    # One task per partition, each reporting its own host. On a laptop this measures the
    # same silicon as the driver and is a control; on a cluster it is the measurement,
    # because the executor is a different machine from the driver and the claim under
    # test is that the ranking does not depend on which machine it is.
    if not args.skip_executor_micro:
        t0 = time.perf_counter()
        dim, sdim, nnz, reps = args.dim, args.sparse_dim, args.nnz, args.repeats

        def run_remote(pid, _it):
            rows = micro_suite(dim, sdim, nnz, reps)
            host = socket.gethostname()
            for section, container, metric, value in rows:
                yield (pid, host, section, container, metric, float(value))

        remote = (sc.parallelize(range(parts), parts)
                    .mapPartitionsWithIndex(run_remote)
                    .collect())

        hosts = sorted({r[1] for r in remote})
        emit("executor_micro", "hosts", "distinct", len(hosts))
        for h in hosts:
            emit("executor_micro", "host", h)

        # One task per partition means `parts` measurements of each number. Report the
        # median across tasks: a single task that landed next to a noisy neighbour is
        # not evidence about the container.
        grouped = {}
        for _pid, _host, section, container, metric, value in remote:
            grouped.setdefault((section, container, metric), []).append(value)
        for (section, container, metric), values in grouped.items():
            arr = np.array(values)
            emit("executor_" + section, container, metric,
                 round(float(np.median(arr)), 4), round(float(arr.min()), 4),
                 round(float(arr.max()), 4), len(arr))
            record("executor_" + section, container, metric, float(np.median(arr)))
        emit("timing", "executor_micro", "seconds", round(time.perf_counter() - t0, 2))

    # ---- part 3: one batch gradient, five containers -------------------------
    # g = sum_i (w.x_i - y_i) x_i — one map (the per-record term) and one treeAggregate
    # (the sum). The per-record term is exactly the dot product of part 1 and the axpy of
    # part 1, run once per row, so this section is the prediction of part 1 at scale.
    # The fifth container leaves the arithmetic to the engine and only exists on Spark
    # 4.2 and later, so it is probed for rather than assumed.
    engine_ok = not args.skip_engine and engine_vector_fns(spark)
    n, d = args.rows, args.features
    w_true = np.random.default_rng(1).standard_normal(d)
    rows = build_rows(sc, n, d, parts, w_true)
    w = np.zeros(d)
    # Computed once here rather than inside the `if engine_ok:` block below, because
    # part 4's residual probe (further down) reuses it too and should not depend on
    # part 3 having run first to define it.
    w_lit = sql_float32_literal(w) if engine_ok else None

    grads, cache_s, grad_s = {}, {}, {}

    # Fairness rule: every variant is cached and materialized before it is timed and
    # unpersisted after, so no variant is timed against another's saved work and only one
    # copy of the data is resident at a time.

    # -- A: a Python list per row ---------------------------------------------
    if not args.skip_list:
        rows_list = rows.map(lambda r: (r[0].tolist(), r[1]))
        cache_s["python_list"] = materialize(rows_list)
        w_l = w.tolist()

        def term_list(row):
            x, y = row
            e = sum(a * b for a, b in zip(w_l, x)) - y
            return [e * a for a in x]

        def add_lists(a, b):
            return [p + q for p, q in zip(a, b)]

        grad_s["python_list"], grads["python_list"] = timed(
            lambda: rows_list.map(term_list)
                             .treeAggregate([0.0] * d, add_lists, add_lists),
            args.gradient_repeats)
        rows_list.unpersist()

    # -- B: a NumPy array per row ---------------------------------------------
    cache_s["numpy"] = materialize(rows)

    def term_np(row):
        x, y = row
        return (x @ w - y) * x

    grad_s["numpy"], grads["numpy"] = timed(
        lambda: rows.map(term_np).treeAggregate(np.zeros(d), np.add, np.add),
        args.gradient_repeats)
    rows.unpersist()

    # -- C: a pyspark.ml DenseVector per row ----------------------------------
    # The wrapper of part 1, now paid once per row. This is the container a `features`
    # column holds, so it is the one that matters.
    rows_ml = rows.map(lambda r: (MLVectors.dense(r[0]), r[1]))
    cache_s["ml_dense"] = materialize(rows_ml)
    w_vec = MLVectors.dense(w)
    zero_vec = MLVectors.dense(np.zeros(d))

    def term_ml(row):
        x, y = row
        return (x.dot(w_vec) - y) * x            # DenseVector in, DenseVector out

    grad_s["ml_dense"], grads["ml_dense"] = timed(
        lambda: rows_ml.map(term_ml).treeAggregate(zero_vec,
                                                   lambda a, b: a + b,
                                                   lambda a, b: a + b),
        args.gradient_repeats)

    # -- D: a features column of a DataFrame ----------------------------------
    # The same DenseVectors, but cached on the JVM side in Spark's columnar format.
    # Running a Python function over it means df.rdd, and df.rdd converts every row to a
    # Python Row and unpickles its vector on *every* pass, because the cached thing is
    # the JVM DataFrame and not the Python objects.
    df = spark.createDataFrame(rows_ml, ["features", "label"])
    cache_s["dataframe_column"] = materialize(df)
    rows_ml.unpersist()

    def term_row(row):
        x = row["features"]
        return (x.dot(w_vec) - row["label"]) * x

    grad_s["dataframe_column"], grads["dataframe_column"] = timed(
        lambda: df.rdd.map(term_row).treeAggregate(zero_vec,
                                                   lambda a, b: a + b,
                                                   lambda a, b: a + b),
        args.gradient_repeats)

    # -- E: the same DataFrame, with the arithmetic left to the engine --------
    # Spark 4.2's vector_* functions make the whole gradient one SQL expression, so no
    # Python worker is opened and the DataFrame's cache is finally the cache that is
    # read. The residual is computed in its own projection: written inside the lambda
    # body it is re-evaluated once per element, which costs 3.7x on this data.
    #
    # vector_inner_product and vector_sum have no pyspark.sql.functions wrapper, so they
    # are reached through call_function rather than F.expr: it takes Columns, not SQL
    # text, so the weight literal and the two expressions below are built once as Column
    # objects and reused across every repeat, instead of being re-parsed from a string on
    # each one. transform is a Column-API function directly, needing neither.
    if engine_ok:
        eng = df.select(vector_to_array("features", dtype="float32").alias("f32"),
                        F.col("label").cast("float").alias("y32"))
        cache_s["engine_vector_fns"] = materialize(eng)
        w_col = float32_array_column(w)
        residual = (F.call_function("vector_inner_product", F.col("f32"), w_col)
                    - F.col("y32")).alias("r")
        gradient = F.call_function(
            "vector_sum",
            F.transform(F.col("f32"), lambda v: (v * F.col("r")).cast("float"))
        ).alias("g")
        grad_s["engine_vector_fns"], g_val = timed(
            lambda: eng.withColumn("r", residual).select(gradient).first()["g"],
            args.gradient_repeats)
        grads["engine_vector_fns"] = np.array(g_val, dtype=np.float64)
        eng.unpersist()
    df.unpersist()

    # -- the agreement check, before any number is read as a comparison --------
    ref = np.asarray(grads["numpy"])
    checks = {
        "ml_dense": np.asarray(grads["ml_dense"].toArray()),
        "dataframe_column": np.asarray(grads["dataframe_column"].toArray()),
    }
    if not args.skip_list:
        checks["python_list"] = np.array(grads["python_list"])
    for name, g in checks.items():
        agree = bool(np.allclose(g, ref))
        emit("gradient_agree", name, agree, round(float(np.max(np.abs(g - ref))), 12))
        if not agree:
            die("gradient from %s disagrees with the NumPy gradient" % name)
    emit("gradient_agree", "numpy", True, 0.0)
    # The engine variant accumulates in float32, so it cannot meet np.allclose's
    # default tolerance. Its agreement is relative, and the error it finds is reported
    # rather than asserted away.
    if "engine_vector_fns" in grads:
        rel = float(np.max(np.abs(grads["engine_vector_fns"] - ref))
                    / max(float(np.max(np.abs(ref))), 1e-12))
        emit("gradient_agree", "engine_vector_fns", bool(rel < 1e-4), "%.3e" % rel)
        record("gradient", "engine_vector_fns", "relative_error", rel)
        if rel >= 1e-4:
            die("engine gradient disagrees with the NumPy gradient")

    base = grad_s["numpy"]
    for name in ("python_list", "numpy", "ml_dense", "dataframe_column",
                 "engine_vector_fns"):
        if name not in grad_s:
            continue
        emit("gradient", name,
             round(cache_s[name], 3),          # seconds to cache: the cost of moving it
             round(grad_s[name], 3),           # seconds for the gradient, best of repeats
             round(grad_s[name] / base, 2),    # against NumPy
             round(grad_s[name] / n * 1e6, 2))  # microseconds per row
        record("gradient", name, "cache_seconds", cache_s[name])
        record("gradient", name, "gradient_seconds", grad_s[name])
        record("gradient", name, "vs_numpy", grad_s[name] / base)
        record("gradient", name, "us_per_row", grad_s[name] / n * 1e6)

    # ---- part 4: the whole vector function family, evaluated by the engine ----
    # Spark 4.2 added seven vector SQL expressions — five scalar ones
    # (vector_inner_product, vector_cosine_similarity, vector_l2_distance, vector_norm,
    # vector_normalize) and two aggregates (vector_sum, vector_avg). They are a fourth
    # answer to "where is the vector stored": not in a Python object at all, but in a
    # float32 array that Catalyst evaluates with no Python worker in the job.
    #
    # Four qualifications, every one of them measured here rather than asserted:
    #   * they are SQL-only. There is no `pyspark.sql.functions.vector_inner_product`,
    #     so they are reached through F.expr;
    #   * they require ARRAY<FLOAT>. A VectorUDT column is rejected outright, so the
    #     features are converted once with vector_to_array(..., dtype="float32");
    #   * float32 is half the precision of every other route in this script, so each
    #     result is compared with its float64 Python equivalent and the relative
    #     difference is printed;
    #   * in 4.2.0 the intermediates accumulate in float32 as well, so these functions
    #     silently return NaN or Infinity for inputs whose true answer is an ordinary
    #     float (SPARK-58544, fixed after 4.2.0). The probe at the end of this section
    #     records what this Spark actually does.
    #
    # This section is skipped where the functions do not exist, which includes Dataproc
    # Serverless runtime 3.0 (Spark 4.0.1). The cluster column of the report is therefore
    # empty here by construction. The gradient itself is not measured here; it is the
    # fifth container of part 3, so that it is compared against the other four on the
    # same rows, with the same repeats and the same agreement check.
    if engine_ok:
        emit("engine", "available", True, spark.version)

        # The five scalar functions are the vector-search family: they compare each row
        # with a query vector. The gradient's w is the zero vector, which makes cosine
        # similarity and normalization undefined, so the query here is the true
        # coefficient vector — an arbitrary non-zero vector of the right width.
        q32 = w_true.astype(np.float32)
        q_lit = sql_float32_literal(q32)
        q_vec = MLVectors.dense(q32.astype(np.float64))
        q_norm = float(np.linalg.norm(q32.astype(np.float64)))

        src = spark.createDataFrame(rows_ml, ["features", "label"])

        # Two cached sources, one per venue of evaluation. The conversion to float32 is
        # itself a cost and is reported: it is paid once, where the Python routes pay
        # their JVM/Python crossing on every pass.
        t0 = time.perf_counter()
        eng = (src.select(vector_to_array("features", dtype="float32").alias("f32"),
                          F.col("label").cast("float").alias("y32"))
                  .cache())
        eng.count()
        emit("engine_prepare", "float32_column_cache", round(time.perf_counter() - t0, 3))
        emit("engine_prepare", "python_rows_cache", round(materialize(rows_ml), 3))

        def engine_scalar(expr):
            """One pass, summing a scalar expression over every row."""
            return lambda: eng.select(F.sum(F.expr(expr))).first()[0]

        def python_scalar(fn):
            """The same pass in Python, over the same rows as DenseVectors."""
            return lambda: rows_ml.map(lambda r: float(fn(r[0]))).sum()

        # -- the five scalar functions, each against its float64 Python equivalent --
        scalar_cases = [
            ("vector_inner_product",
             "vector_inner_product(f32, %s)" % q_lit,
             lambda x: x.dot(q_vec)),
            ("vector_cosine_similarity",
             "vector_cosine_similarity(f32, %s)" % q_lit,
             lambda x: x.dot(q_vec) / (float(np.linalg.norm(x.toArray())) * q_norm)),
            ("vector_l2_distance",
             "vector_l2_distance(f32, %s)" % q_lit,
             lambda x: math.sqrt(x.squared_distance(q_vec))),
            ("vector_norm",
             "vector_norm(f32)",
             lambda x: np.linalg.norm(x.toArray())),
            ("vector_normalize",
             "element_at(vector_normalize(f32), 1)",
             lambda x: x[0] / float(np.linalg.norm(x.toArray()))),
        ]
        for name, expr, pyfn in scalar_cases:
            t_e, v_e = timed(engine_scalar(expr), args.gradient_repeats)
            t_p, v_p = timed(python_scalar(pyfn), args.gradient_repeats)
            # A NULL is a result too — vector_cosine_similarity returns one for a
            # zero-magnitude vector — so it is reported rather than allowed to crash.
            rel = ("null" if v_e is None
                   else "%.3e" % (abs(v_e - v_p) / max(abs(v_p), 1e-12)))
            emit("engine_fn", name, round(t_e, 3), round(t_p, 3),
                 round(t_p / t_e, 2), round(t_e / n * 1e6, 3), rel)
            record("engine_fn", name, "engine_seconds", t_e)
            record("engine_fn", name, "python_seconds", t_p)
            record("engine_fn", name, "speedup", t_p / t_e)
            if rel != "null":
                record("engine_fn", name, "relative_error", float(rel))

        # -- the two aggregates, which return a vector rather than a scalar --------
        agg_cases = [
            ("vector_sum", "vector_sum(f32)", 1.0),
            ("vector_avg", "vector_avg(f32)", float(n)),
        ]
        t_p_sum, v_p_sum = timed(
            lambda: rows_ml.map(lambda r: r[0].toArray())
                           .treeAggregate(np.zeros(d), np.add, np.add),
            args.gradient_repeats)
        for name, expr, divisor in agg_cases:
            t_e, v_e = timed(
                lambda e=expr: eng.select(F.expr(e).alias("v")).first()["v"],
                args.gradient_repeats)
            ref_v = v_p_sum / divisor
            rel = float(np.max(np.abs(np.array(v_e, dtype=np.float64) - ref_v))
                        / max(float(np.max(np.abs(ref_v))), 1e-12))
            emit("engine_fn", name, round(t_e, 3), round(t_p_sum, 3),
                 round(t_p_sum / t_e, 2), round(t_e / n * 1e6, 3), "%.3e" % rel)
            record("engine_fn", name, "engine_seconds", t_e)
            record("engine_fn", name, "python_seconds", t_p_sum)
            record("engine_fn", name, "speedup", t_p_sum / t_e)
            record("engine_fn", name, "relative_error", rel)

        # -- where the residual is written, which is the whole cost -------------
        # The gradient itself is part 3's fifth container. What belongs here is the one
        # thing that section cannot show: the same expression with the residual moved
        # inside the lambda body, where Catalyst re-evaluates it once per element
        # instead of once per row.
        inline = F.expr(
            "vector_sum(transform(f32, v -> cast(v * (vector_inner_product(f32, %s)"
            " - y32) as float)))" % w_lit)
        t_in, _ = timed(lambda: eng.select(inline.alias("g")).first()["g"],
                        args.gradient_repeats)
        hoisted = grad_s.get("engine_vector_fns")
        emit("engine_residual", "inside_lambda_body", round(t_in, 3),
             round(hoisted, 3) if hoisted else "-",
             round(t_in / hoisted, 2) if hoisted else "-")
        record("engine_residual", "inside_lambda_body", "seconds", t_in)

        # -- what this Spark does with SPARK-58544's inputs ------------------------
        # Every expected value below is an ordinary float; only the float32 intermediate
        # overflows. Printed rather than asserted, so the line reads as a record of this
        # version's behaviour on any Spark the script is run on.
        probes = [
            ("cosine_big", "vector_cosine_similarity(array(3.0e19F,4.0e19F),"
                           " array(3.0e19F,4.0e19F))", "1.0"),
            ("l2_big", "vector_l2_distance(array(3.0e19F,4.0e19F), array(0.0F,0.0F))",
             "5.0E19"),
            ("norm_big", "vector_norm(array(3.0e19F,4.0e19F), 2.0F)", "5.0E19"),
            ("cosine_small", "vector_cosine_similarity(array(1.0e-23F,0.0F),"
                             " array(1.0e-23F,0.0F))", "1.0"),
        ]
        for label, expr, expected in probes:
            try:
                got = spark.sql("SELECT %s AS v" % expr).first()["v"]
            except Exception as exc:
                got = "error: %s" % str(exc).split("\n")[0][:60]
            emit("engine_overflow", label, got, expected)

        eng.unpersist()
        rows_ml.unpersist()
    else:
        # Recorded as a measurement, not a silence: the report's cloud column is empty
        # here because the functions do not exist there, and the log says so.
        emit("engine", "available", False, spark.version)
    if args.out:
        out = spark.createDataFrame(ROWS, ["section", "container", "metric", "value"])
        out.coalesce(1).write.mode("overwrite").option("header", True).csv(args.out)
        emit("wrote", args.out)

    emit("done")
    spark.stop()


if __name__ == "__main__":
    main()
