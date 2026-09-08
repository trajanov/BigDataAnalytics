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
