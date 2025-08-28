# GDELT https://www.gdeltproject.org/ is a global event database that extracts events from news articles around the world
# The dataset is updated every 15 minutes and is available in CSV format
# The dataset contains information about the events such as the date, actors involved, location, and more
# The dataset is available at http://data.gdeltproject.org/gdeltv2/lastupdate.txt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, desc, count, sum
from pyspark.sql.types import StructType, StructField, StringType, TimestampType

# set  WARN level to ERROR to avoid excessive logging
import logging
logging.getLogger("py4j").setLevel(logging.ERROR)
# set WARN level to ERROR to avoid excessive logging
logging.getLogger("pyspark").setLevel(logging.ERROR)
# set WARN level to ERROR to avoid excessive logging

# Initialize Spark session

spark = (
    SparkSession 
    .builder 
    .appName("GDELT CSV Streaming Processing") 
    .config("spark.sql.adaptive.enabled", "false") \
    .config("spark.sql.shuffle.partitions", 8)
    .master("local[*]") 
    .getOrCreate()
)

# Define schema for GDELT CSV files
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
    StructField("Actor1Geo_Lat", StringType(), True),
    StructField("Actor1Geo_Long", StringType(), True),
    StructField("Actor1Geo_FeatureID", StringType(), True),
    StructField("Actor2Geo_Type", StringType(), True),
    StructField("Actor2Geo_FullName", StringType(), True),
    StructField("Actor2Geo_CountryCode", StringType(), True),
    StructField("Actor2Geo_ADM1Code", StringType(), True),
    StructField("Actor2Geo_Lat", StringType(), True),
    StructField("Actor2Geo_Long", StringType(), True),
    StructField("Actor2Geo_FeatureID", StringType(), True),
    StructField("ActionGeo_Type", StringType(), True),
    StructField("ActionGeo_FullName", StringType(), True),
    StructField("ActionGeo_CountryCode", StringType(), True),
    StructField("ActionGeo_ADM1Code", StringType(), True),
    StructField("ActionGeo_Lat", StringType(), True),
    StructField("ActionGeo_Long", StringType(), True),
    StructField("ActionGeo_FeatureID", StringType(), True),
    StructField("DATEADDED", StringType(), True),
    StructField("SOURCEURL", StringType(), True)
])

# Define the source of the streaming data
input_path = "input_files"  # Path to GDELT CSV data file

# Read streaming data from CSV files
df = spark.readStream \
    .option("header", "false") \
    .option("delimiter", "\t") \
    .schema(schema) \
    .csv(input_path)

# Convert GoldsteinScale to float
df = df.withColumn("GoldsteinScale", col("GoldsteinScale").cast("float"))

# Load country code mapping file
country_mapping_path = "CAMEO.country.txt"  # Path to country mapping CSV file
country_schema = StructType([
    StructField("CountryCode", StringType(), True),
    StructField("CountryName", StringType(), True)
])

country_mapping_df = spark.read \
    .option("header", "true") \
    .option("delimiter", "\t") \
    .schema(country_schema) \
    .csv(country_mapping_path)

# Join the GDELT data with the country mapping to get country names
joined_df = df.join(country_mapping_df, df.Actor1CountryCode == country_mapping_df.CountryCode, "left")

# Group by country name and calculate the average of GoldsteinScale and count of events
country_goldstein_df = joined_df.groupBy("CountryName") \
    .agg(
        avg("GoldsteinScale").alias("AverageGoldsteinScale"),
        count("GLOBALEVENTID").alias("NumberOfEvents"),
    )

# Get the top 10 most positive and 10 most negative countries
most_positive_countries = country_goldstein_df.orderBy(desc("AverageGoldsteinScale")).limit(10)
most_negative_countries = country_goldstein_df.orderBy("AverageGoldsteinScale").limit(10)

# top 10 countries with the highest number of events
most_events_countries = country_goldstein_df.orderBy(desc("NumberOfEvents")).limit(10)

# Write the results to the console in real-time
query_positive = most_positive_countries.writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("truncate", "false") \
    .start()

query_negative = most_negative_countries.writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("truncate", "false") \
    .start()
    #.option("checkpointLocation", "checkpoint") \

query_events = most_events_countries.writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("truncate", "false") \
    .start()
    
query_positive.awaitTermination()
query_negative.awaitTermination()
query_events.awaitTermination()
