# CS-777 Chapter 9 — the socket word count.
#
# Two terminals. In the first, a TCP server that forwards whatever you type:
#
#     nc -lk 9999
#
# In the second, this application:
#
#     .venv/bin/spark-submit "code/scripts/09.02 Socket Word Count.py"
#
# Type words into the Netcat window and the counts appear here, revised after
# every micro-batch. Stop with Ctrl-C.
#
# The socket source is for demonstrations only. It has no replayable offsets, so
# it cannot recover after a failure, and its listening server lives on the driver.

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split

spark = SparkSession \
    .builder \
    .appName("StructuredNetworkWordCount") \
    .config("spark.sql.adaptive.enabled", "false") \
    .config("spark.sql.shuffle.partitions", 8)\
    .getOrCreate()

# Create DataFrame representing the stream of input lines from connection to localhost:9999
lines = spark \
    .readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# Split the lines into words
words = lines.select(explode(split(lines.value, " ")).alias("word"))

# Generate running word count
wordCounts = words.groupBy("word").count()

# Start running the query that prints the running counts to the console
query = wordCounts \
    .writeStream \
    .outputMode("update") \
    .format("console") \
    .start()

query.awaitTermination()
