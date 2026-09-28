# `code/cloud/` — scripts submitted to a cloud Spark service

A notebook is the wrong vehicle for a serverless submission: the services take a **script**,
run it, and destroy the machines. What is here is the script form of work whose point is
that it runs somewhere other than a laptop.

Filenames follow the same `CC.NN Title.ext` convention as `code/notebooks/`, so they carry
spaces. **Quote every path**; the commands below already do.

The account setup, bucket creation, permissions, and the console equivalent of each command
are in the cloud appendix of the lecture notes
(`LectureNotes/chapters/appendix-cloud.tex`). They are deliberately not repeated here:
provider interfaces change and the appendix is the one place that tracks them.

---

## `01.04 Word Count Cloud Job.py`

**Chapter 1, Exercise 8** — *word count at two scales*. Run it locally, then run the same
pipeline against a several-hundred-megabyte corpus in object storage, submitted as a
serverless job, and account for the difference.

The pipeline is identical to the one in `code/notebooks/01.02 Word Count.ipynb`, so the only
variables in the experiment are the size of the input and where the machines are.

### 1. Run it locally

```bash
source spark-env.sh
rm -rf /tmp/cs777/wordcount-local
.venv/bin/spark-submit "code/cloud/01.04 Word Count Cloud Job.py" \
    code/data/Alices-Adventures-in-Wonderland-by-Lewis-Carroll.txt.bz2 \
    /tmp/cs777/wordcount-local
```

The output directory must **not** already exist; Spark refuses to overwrite one.

Record the `ELAPSED_SECONDS=` line.

### 2. Get a corpus of a few hundred megabytes

The shipped corpus is 49 KB, which is far too small for the exercise: at that size you would
be timing startup and nothing else — which is, admittedly, the lesson. Build a larger one by
concatenating a public-domain text with itself until it is big enough:

```bash
bzcat code/data/Alices-Adventures-in-Wonderland-by-Lewis-Carroll.txt.bz2 > /tmp/alice.txt
for i in $(seq 1 2000); do cat /tmp/alice.txt; done > /tmp/alice-large.txt
ls -lh /tmp/alice-large.txt          # ~320 MB
```

Plain text rather than `.gz`: **gzip is not splittable**, so however large a `.gz` file is,
Spark reads it with exactly one task and the run tells you nothing about parallelism. Use
plain text, or `bzip2`, which is splittable.

### 3. Upload the corpus and the script

```bash
gsutil cp /tmp/alice-large.txt                        gs://YOUR-BUCKET/corpus.txt
gsutil cp "code/cloud/01.04 Word Count Cloud Job.py"  "gs://YOUR-BUCKET/01.04 Word Count Cloud Job.py"
```

On AWS, `aws s3 cp` and an `s3://` URI instead.

### 4. Submit it as a serverless job

```bash
gcloud dataproc batches submit pyspark \
    "gs://YOUR-BUCKET/01.04 Word Count Cloud Job.py" \
    --region=us-east1 \
    --version=2.2 \
    -- gs://YOUR-BUCKET/corpus.txt gs://YOUR-BUCKET/wordcount-out/
```

Everything after the bare `--` reaches the script as its arguments, in order. In the web
console's *Create batch* form the same two values go in the **Arguments** field, one chip
each, input first — entering them the other way round is the usual way a first submission
goes wrong.

### 5. Recover the number

A serverless batch leaves **no cluster and no Spark web interface behind**, which is exactly
the practical consequence §1.12 of the notes warns about. The script therefore prints its
timing in a form you can grep out of the driver log:

```bash
gcloud dataproc batches describe BATCH_ID --region=us-east1
# then open the driver output, or:
gcloud logging read 'resource.type="cloud_dataproc_batch"' --limit=200 \
    | grep ELAPSED_SECONDS
```

### What to expect

The cloud run will very likely be **slower** than the local one, and on a small file it will
be dramatically slower. That is the point of the exercise. The time is going to job
submission, machine allocation, container and JVM startup, and object-store latency — none of
which is the word count. Write the two numbers down and account for the gap; a distributed
system is not a faster computer, it is a larger one, and you pay for the size whether or not
you need it.

---

## `06.06 Vector Storage Benchmark.py`

**Chapter 6, §§6.1.1, 6.1.5 and 6.2.1** — *what the container a vector is stored in costs*.
The script form of `code/notebooks/06.06 Vector Storage Compared.ipynb`.

It reads **no input**. Every vector is synthesized from a fixed seed and the rows of the
distributed section are generated inside the tasks, so a cluster run and a laptop run
measure the same computation.

### What it adds to the notebook

The notebook's claim is that the ranking of containers — Python list, NumPy array, SciPy
sparse matrix, `pyspark.ml` vector, `pyspark.mllib` vector, DataFrame column — is the same
on the driver as inside a task. A laptop cannot test the second half of that, because there
the driver and the task are the same machine. The script runs the identical micro-suite in
three places:

1. on the **driver**;
2. inside a **Python worker on every partition**, reporting the median, minimum and maximum
   across tasks, together with each task's host name;
3. across a whole **distributed batch gradient**, four containers, each cached and
   materialized before it is timed and unpersisted after.

Every result line is prefixed `RESULT` and tab-separated, because a serverless batch leaves
no notebook and no Spark UI behind and the driver log is the whole record of the run.

### 1. Run it locally first

```bash
source spark-env.sh
"$SPARK_HOME/bin/spark-submit" --master "local[4]" \
    "code/cloud/06.06 Vector Storage Benchmark.py" \
    --rows 100000 --features 200 --partitions 4 --repeats 5 --gradient-repeats 3 \
    > code/gcp/results/local-0606/driver.log 2>&1
```

### 2. Submit it to Dataproc Serverless

```bash
code/gcp/run_bench_0606.sh                       # the same shape as the laptop run
code/gcp/run_bench_0606.sh 1000000 200           # rows features
BENCH_ARGS="--skip-list" code/gcp/run_bench_0606.sh    # drop the slow variant
```

The runner sources `code/gcp/config.sh` and `auth.sh`, uploads the script, submits the
batch, and saves `config.txt`, `driver.log`, `batch.json` and — when the streamed log comes
back without the result lines, which is usual — `logging.log` read out of Cloud Logging,
under `code/gcp/results/BATCH_ID/`.

### 3. Turn the two logs into one report

```bash
.venv/bin/python3 code/gcp/report_bench_0606.py \
    code/gcp/results/local-0606/driver.log \
    code/gcp/results/BATCH_ID/logging.log \
    -o code/gcp/results/06.06-local-vs-cloud.md
```

The findings are generated from the logs along with the tables, so re-running the benchmark
rewrites the conclusions and no sentence can quote a number the current run did not
produce. A committed example is `code/gcp/results/06.06-local-vs-cloud.md`.

The narrative write-up of the 2026-09-19 pair of runs — the setup, the local results and the
cloud results, each explained — is `code/gcp/results/06.06-experiment-report.md`. The generated
file holds the tables; that one holds the reading of them.

### What to expect

The ranking survives the trip and the absolute numbers do not. A cloud core is several times
slower than a current laptop core at the same NumPy dot product, and at a hundred thousand
rows the whole gradient is slower on the cluster than on the laptop — there is nothing at
that size for a cluster to do that one machine cannot, and what the cluster adds is
scheduling, network serialization and slower cores.

One instrument note: on Dataproc Serverless every task reports the same
`socket.gethostname()`, because the containers are named after the runtime manager node. The
per-task timings are executor-side measurements; what they cannot do is tell one executor
from another.

---

## A note on paths with spaces

`spark-submit` resolves `SPARK_HOME` through a shell script that does not quote the path it
finds. If the repository lives somewhere with a space in it — a Google Drive folder, for
instance — `spark-submit` fails with

```
find-spark-home: line 40: /Users/you/My: No such file or directory
```

The script itself is unaffected; only the launcher is. Either clone the repository to a path
with no spaces, or set `SPARK_HOME` explicitly and invoke the launcher from there:

```bash
SH="$(.venv/bin/python3 -c 'import pyspark, os; print(os.path.dirname(pyspark.__file__))')"
SPARK_HOME="$SH" "$SH/bin/spark-submit" "code/cloud/01.04 Word Count Cloud Job.py" IN OUT
```

This does not arise on the cloud services, which supply their own Spark installation.

---

## `08.08 DataFrame vs RDD Benchmark.py`

**Chapter 8, §8.1.1's note box** — *the hand-written gradient loop on RDDs against the same
loop written relationally*. The script form of
`code/notebooks/08.08 DataFrame Versions and Their Speed.ipynb`, so the comparison can be
made somewhere other than a laptop.

It reads **no input**. The data is synthesized from a fixed seed, so a cluster run and a
local run measure the same computation and the numbers are comparable.

### What it does

Four training loops — linear and logistic regression, each written once on an RDD and once
as a single `agg` over a DataFrame — timed across a list of row counts. Before anything is
timed, the two representations of each model are asserted to produce the same parameter
vector, so a speed difference can never be a difference in what was computed. It then swaps
the settings that differ between the two APIs: `reduce` against `treeAggregate` on the RDD
side, and adaptive execution and whole-stage code generation on the DataFrame side.

### Two differences from the notebook, both forced by the venue

* **The data is generated on the executors**, by a `flatMap` over partition indices with a
  per-partition seed, rather than built on the driver and shipped out by `sc.parallelize`.
  Sending millions of NumPy arrays out of a driver becomes the dominant cost long before
  the gradient loop does, and it is not the cost under study.
* **It prints rather than plots.** A serverless batch leaves no notebook and no Spark UI
  behind, so the driver log is the whole record of the run. Every result line is prefixed
  `RESULT` and tab-separated so it can be recovered from a log containing nothing else.

### 1. Run it locally first

```bash
source spark-env.sh
spark-submit "code/cloud/08.08 DataFrame vs RDD Benchmark.py" --sizes 25000,100000 --repeats 2
```

If `spark-submit` fails on a path containing a space, see the note at the foot of this file.

### 2. Submit it to Dataproc Serverless

```bash
code/gcp/run_bench_0808.sh                            # default sizes
code/gcp/run_bench_0808.sh 100000,1000000,10000000    # sizes worth a cluster
code/gcp/run_bench_0808.sh 1000000 20 3 64            # sizes iters repeats partitions
```

The runner sources `code/gcp/config.sh` for the project, region, bucket and runtime, uploads
the script, submits the batch, and greps the driver log for the `RESULT` lines. Options
reach the script after the bare `--`, which the runner supplies.

### 3. Recover the numbers by hand, if the runner cannot

```bash
gcloud logging read \
  'resource.type="cloud_dataproc_batch" AND labels."dataproc.googleapis.com/batch_id"="BATCH_ID"' \
  --limit 500 --format='value(textPayload)' | grep RESULT
```

### What to expect

Locally the two curves cross at around fifty thousand rows: below it the RDD loop is faster,
above it the relational one, by a margin that widens with the data. A cluster should move
that crossing point, because the RDD's dominant cost is per-record Python serialization,
which divides across executors, while the DataFrame's is per-iteration query planning on the
driver, which does not. **That is a prediction and not yet a measurement** — see the status
note in §5e of the notebook.

---

## Authentication, and why the login cannot be scripted

`code/gcp/auth.sh` is sourced by the runners. It uses a service account when its key is
present and the active user credential otherwise, and it fails early with a legible message
when neither works.

The distinction matters because of a failure that looks like an expired token and is not:

```
ERROR: There was a problem refreshing your current auth tokens: Reauthentication failed.
cannot prompt during non-interactive execution.
```

That is Google's **reauth challenge**, which a Workspace administrator configures through
Google Cloud session control. Its purpose is to confirm a human is present, so no flag or
configuration makes a personal account pass it unattended. `--impersonate-service-account`
does not help either, because impersonation still needs a valid base credential.

The fix is a different identity rather than a different flag. A service account's key mints
short-lived tokens automatically and is never challenged. The one-time setup is written out
at the top of `code/gcp/auth.sh`; after it, every run authenticates with no prompt.

Two things can stop that. Organizations frequently disable key creation through the policy
`constraints/iam.disableServiceAccountKeyCreation`, in which case laptop-local automation is
not available at all. And a key is a long-lived secret: it belongs under `$HOME`, never in
this directory, which is why `auth.sh` looks for it at
`~/.config/gcloud/keys/cs777-batch.json` and takes an override through `CS777_GCP_KEY`.

Where the submission does not have to run from a laptop, the simpler answer is to run it from
inside GCP. Cloud Shell, a Compute Engine VM, and Cloud Build each carry an attached service
account, need no login, and leave no long-lived secret anywhere.
