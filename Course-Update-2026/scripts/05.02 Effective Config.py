"""CS-777 5.2 — what configuration did this application actually receive?

A deliberately minimal Spark application whose only job is to report its own settings. It is
the Environment tab of the Spark UI, printed to standard output, and it exists so that the
precedence rules of chapter 5 section 5.3 can be demonstrated rather than described: the same
script is submitted several times with the same code and different submission options, and the
values it reports are the evidence.

Run it the way the chapter runs any Python application:

    spark-submit "05.02 Effective Config.py"
    spark-submit --conf spark.sql.shuffle.partitions=240 "05.02 Effective Config.py"
    spark-submit --executor-memory 3g --conf spark.sql.shuffle.partitions=240 \
                 "05.02 Effective Config.py"
    spark-submit --properties-file my-defaults.conf "05.02 Effective Config.py"

Any property names given as arguments after the script are reported in addition to the standard
set, which is the third group of options in section 5.4: everything after the script name is
passed to the program, not to Spark.

Notebook `05.02 Configuration and Precedence.ipynb` submits this script and reads its output.
"""

import os
import sys
import tempfile

from pyspark.sql import SparkSession

# The properties the chapter discusses, in the order it discusses them. Two are launch-time
# and two are runtime-modifiable, which is the distinction the notebook is built around.
REPORTED = [
    "spark.app.name",
    "spark.master",
    "spark.submit.deployMode",
    "spark.driver.memory",
    "spark.executor.memory",
    "spark.executor.cores",
    "spark.sql.shuffle.partitions",
    "spark.sql.files.maxPartitionBytes",
]


def main(extra):
    scratch = os.environ.get(
        "CS777_SCRATCH", os.path.join(tempfile.gettempdir(), "cs777"))
    os.makedirs(scratch, exist_ok=True)

    # Two properties are set here, in the program, which the chapter calls the highest of the
    # three precedences. Whether each one wins is the whole point of submitting this more than
    # once, and the two behave differently:
    #
    #   maxPartitionBytes is runtime-modifiable, so setting it here is honoured.
    #   driver.memory is launch-time, so under spark-submit the driver JVM already exists and
    #   this line configures nothing -- while still reporting itself as set. The heap figure
    #   printed at the end of this script is the only thing that says which happened.
    spark = (SparkSession.builder
             .appName("CS777-5.2-submitted")
             .config("spark.sql.files.maxPartitionBytes", "64m")
             .config("spark.driver.memory", "3g")
             .config("spark.ui.showConsoleProgress", "false")
             .config("spark.sql.warehouse.dir", os.path.join(scratch, "warehouse"))
             .getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    print("CS777-CONFIG-BEGIN")
    for key in REPORTED + list(extra):
        try:
            value = spark.conf.get(key)
        except Exception:
            value = "<not set>"
        modifiable = spark.conf.isModifiable(key)
        print(f"CS777-CONFIG {key}={value} modifiable={modifiable}")

    # The JVM's own view of the driver heap. This is the only honest way to tell whether
    # spark.driver.memory did anything, because the property is reported whether or not the
    # JVM was ever launched with it.
    heap_mb = sc._jvm.java.lang.Runtime.getRuntime().maxMemory() / (1024 * 1024)
    print(f"CS777-CONFIG driver.jvm.maxHeapMiB={heap_mb:.0f} modifiable=n/a")
    print("CS777-CONFIG-END")

    spark.stop()


if __name__ == "__main__":
    main(sys.argv[1:])
