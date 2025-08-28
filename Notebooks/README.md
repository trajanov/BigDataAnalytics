# Learning PySpark by examples

*By: Dimitar Trajanov*

Learning PySpark by examples can be an effective way to understand how PySpark works in practice and to gain hands-on experience with the tool. Here are a few reasons why learning PySpark by examples can be beneficial:

1. Demonstrates real-world applications: Examples show how PySpark is used to solve real-world data processing challenges. This can help you understand how to apply PySpark to your own data analysis problems.
2. Provides practical experience: Working through examples can help you gain practical experience with PySpark. By experimenting with PySpark code, you can better understand how to use PySpark functions, manipulate data, and build data pipelines.
3. Improves problem-solving skills: Examples can also help you develop your problem-solving skills. As you work through examples, you will encounter errors and challenges that you will need to solve. This can help you develop your critical thinking skills and learn how to debug PySpark code.
4. Learning PySpark by examples can also be a fun and engaging way to learn. Examples can help make learning more interactive and hands-on, which can be more enjoyable than reading through documentation or tutorials.

In this repository, you can find examples for PySpark that range from the basics all the way up to advanced machine learning algorithms, including those using MLLib. These examples provide a comprehensive overview of PySpark's capabilities, from simple data processing tasks to more complex use cases involving advanced machine learning techniques. Whether you are just starting out with PySpark or are already experienced with the tool, you can find examples that will help you build your skills and develop your understanding of how to use PySpark to solve real-world data problems.

## PySpark official documentation

To help users get started with PySpark, the official documentation provides a comprehensive guide that covers all aspects of the framework, from installation to advanced data processing and machine learning algorithms.

The official PySpark documentation is maintained by the developers themselves, ensuring that it is always up-to-date and accurate with the latest changes in the platform.

The following guides are highly recommended for anyone learning or working with PySpark:

- [PySpark Getting Started](https://spark.apache.org/docs/latest/api/python/getting_started/index.html): This is the Getting Started guide that summarizes the basic steps required to setup and gets started with PySpark.
- [RDD Programming Guide](https://spark.apache.org/docs/latest/rdd-programming-guide.html): This guide provides an overview of Spark basics, including RDDs (the core but older API), accumulators, and broadcast variables.
- [Spark SQL, Datasets, and DataFrames](https://spark.apache.org/docs/latest/sql-programming-guide.html): This guide is focused on processing structured data with relational queries using a newer API than RDDs.
- [MLlib](https://spark.apache.org/docs/latest/ml-guide.html): This guide provides detailed information on how to apply machine learning algorithms in PySpark.
- [Spark Python API](https://spark.apache.org/docs/latest/api/python/reference/index.html). This is the PySpark API documentation, which provides detailed information on the PySpark API, including its modules, classes, and methods.
---
# Notebooks

## Section 1 — Getting started

- [Spark-Example-01-Word-Count-Example.ipynb](Spark-Example-01-Word-Count-Example.ipynb) — Classic word-count using RDDs with several incremental cells: basic map/flatMap/reduceByKey flow, a top-k example, and an "optimized" pipeline that lowercases and strips punctuation before aggregation. Good first notebook to verify SparkSession setup and learn the RDD action/transformation cycle. Run notes: includes in-notebook sample lines and demonstrates collecting vs distributed actions.

- [Spark-Example-02-RDD Basics Toutorial.ipynb](Spark-Example-02-RDD%20Basics%20Toutorial.ipynb) — A thorough, teaching-focused tour of the RDD API: parallelize, collect/take, map/filter/reduce, persistence, and visual diagrams. Contains annotated notes and examples that explain why operations are lazy and how partitioning affects parallelism. Run notes: many explanatory markdown cells and small runnable examples; useful as a reference while doing other RDD notebooks.

- [Spark-Example-03-PySpark vs Python.ipynb](Spark-Example-03-PySpark%20vs%20Python.ipynb) — Hands-on comparisons of equivalent algorithms implemented in plain Python (lists/pandas) and in PySpark (RDDs/DataFrames) — examples include sum, word-count, max, and more. Teaches overhead trade-offs and when distributed execution makes sense. Run notes: includes timing and profiling examples; run locally for fair comparisons.

- [Spark-Example-04-Not to use transformations.ipynb](Spark-Example-04-Not%20to%20use%20transformations.ipynb) — Focused on anti-patterns: explains why groupByKey and similar transformations can cause huge shuffles and memory pressure and shows better alternatives. Strongly recommended after learning basic transformations so students learn to avoid expensive patterns.

## Section 2 — Real-world data processing

- [Spark-Example-05-Flight-dataset analysis.ipynb](Spark-Example-05-Flight-dataset%20analysis.ipynb) — End-to-end example for downloading, loading and preprocessing a large flights CSV (links included). Covers removing headers, parsing CSVs compressed with bzip2, joining with airlines and airports lookup files, and computing aggregated metrics (delays, airport stats). Run notes: demonstrates how to fetch sample data (wget in notebook), and shows common CSV ingestion and cleaning steps for large files.

- [Spark-Example-06-Join Operation on RDD.ipynb](Spark-Example-06-Join%20Operation%20on%20RDD.ipynb) — Demonstrates join variants on RDDs (join, leftOuterJoin, rightOuterJoin, fullOuterJoin) and how zipping and co-partitioning change behavior. Good for understanding shuffle costs and the semantics of each join type. Run notes: contains simple synthetic examples to visualize join outputs.

- [Spark-Example-07-RDD-Dataframe-Examples-Operations.ipynb](Spark-Example-07-RDD-Dataframe-Examples-Operations.ipynb) — Collection of canonical RDD examples (reduceByKey, distinct, join) and practical conversions to DataFrames. Shows when schema-aware APIs (DataFrames) simplify code and improve performance. Run notes: includes examples of aggregateByKey and why reduceByKey/aggregateByKey are preferred over naive set/list aggregations.

## Section 3 — Advanced RDD patterns and partitioning

- [Spark-Example-08-PySpark and NumPy.ipynb](Spark-Example-08-PySpark%20and%20NumPy.ipynb) — Explains how to use NumPy inside mapPartitions and UDFs for efficient vectorized per-partition computation, and when to broadcast small arrays. Useful when migrating numerical Python code to distributed execution.

- [Spark-Example-09-Data-Partitioning-TreeAggregate.ipynb](Spark-Example-09-Data-Partitioning-TreeAggregate.ipynb) — Conceptual and practical guide to partitioning strategies (hash, range, custom) and to treeAggregate/treeReduce for lowering network overhead in large reductions. Contains formulas, best practices, and code demonstrating repartitioning and glom inspection.

- [Spark-Example-09.1-treeAggregate-Min-Max-Calculation.ipynb](Spark-Example-09.1-treeAggregate-Min-Max-Calculation.ipynb) — Implements custom seqOp/combOp functions to compute min/max (and keys) in a single pass using treeAggregate; includes numeric-stability notes and test examples on small taxi CSV samples.

## Section 4 — DataFrames, SQL and complex types

- [Spark-Example-10-DataFrames.ipynb](Spark-Example-10-DataFrames.ipynb) — Intro to SparkSession and DataFrame APIs: createDataFrame from lists/RDDs, schema handling, reading/writing Parquet and CSV, column expressions and common transformations. Contains conceptual notes on createDataFrame parameters and examples demonstrating schema inference and verification. Run notes: good primer before moving to SQL-style notebooks.

- [Spark-Example-10.2-Working-With-Dataframes-Array.ipynb](Spark-Example-10.2-Working-With-Dataframes-Array.ipynb) — Shows how to handle array/struct columns, convert Python lists to Spark array types via UDFs, use higher-order functions and UDFs for per-row array math (e.g., prepare feature vectors for gradient descent). Run notes: includes sample UDFs that convert between Python/NumPy types and Spark arrays.

- [Spark-Example-11-Dataframes-Join.ipynb](Spark-Example-11-Dataframes-Join.ipynb) — Demonstrates DataFrame join strategies, aliasing to avoid ambiguous column references, broadcast joins, and inspecting physical plans. Helpful examples for common DataFrame join gotchas and how to produce deterministic results.

- [Spark-Example-12-Taxi Dataset Data Frame Example.ipynb](Spark-Example-12-Taxi%20Dataset%20Data%20Frame%20Example.ipynb) — End-to-end DataFrame pipeline using an NYC taxi sample: loading CSVs, data cleaning (row corrections), timestamp parsing, window functions, and converting to Pandas for small-scale visualization. Run notes: includes helper functions for row validation and demonstrates conversion to pandas when dataset fits memory.

## Section 5 — Aggregations and key-based operations

- [Spark-Example-13-AggregateByKey.ipynb](Spark-Example-13-AggregateByKey.ipynb) — Practical examples of aggregateByKey patterns (seqOp/combOp) to build efficient per-key aggregations without excessive object creation. Shows why aggregateByKey is preferred for complex combiners.

- [Spark-Example-14-combinedByKey.ipynb](Spark-Example-14-combinedByKey.ipynb) — Walks through reduceByKey, aggregateByKey and combineByKey, including building combiners for top-k/sessionization and the three-function contract required by combineByKey.

## Section 6 — Control flow, safe patterns and joins

- [Spark-Example-15-forloop-Variable.ipynb](Spark-Example-15-forloop-Variable.ipynb) — Demonstrates closure-related bugs and shows correct patterns for parameterizing repeated Spark jobs (avoid mutating outer variables inside transformations). Useful for running experiments safely in loops.

- [Spark-Example-16-Map-Side-Join.ipynb](Spark-Example-16-Map-Side-Join.ipynb) — Teaches map-side (broadcast) joins when one side fits in memory: collectAsMap, broadcast, and mapping on the large RDD to avoid a full shuffle. Practical recipe for small-master/large-detail joins.

## Section 7 — Machine learning from first principles

- [Spark-Example-17-Simple-Linear-Regression-Implementation.ipynb](Spark-Example-17-Simple-Linear-Regression-Implementation.ipynb) — Implements simple linear regression from first principles (generate synthetic data, compute loss and slope, compare with sklearn). Great for building intuition about gradients and model fitting before scaling to distributed training.

- [Spark-Example-18-Linear-Regression-Gradient Descent.ipynb](Spark-Example-18-Linear-Regression-Gradient%20Descent.ipynb) — Implements batch/stochastic gradient-descent variants for linear regression, includes data generation, 3D visualizations for multiple features, and reproducible seeds for experiments. Shows how to wire up iterative updates with RDD-based mini-batching.

- [Spark-Example-18-Regularization.ipynb](Spark-Example-18-Regularization.ipynb) — Explores L1/L2 regularization and cross-validation patterns for model selection; complements the gradient-descent notebooks with practical model-selection notes.

- [Spark-Example-19-Logistic-Regression.ipynb](Spark-Example-19-Logistic-Regression.ipynb) — Theoretical and practical walkthrough of logistic regression: Bernoulli likelihood, gradient derivation, synthetic data generation and model training. Includes evaluation metrics (ROC, AUC, precision/recall) and visualizations for classification.

- [Spark-Example-19b-SVM.ipynb](Spark-Example-19b-SVM.ipynb) — SVM implementation using a weighted SGD approach, with loss and gradient formulas included. Good for students who want to inspect optimization details and implement SGD-based learners on RDDs.

- [Spark-Example-20-Adam-Sgdm-with-Tree-Aggregate.ipynb](Spark-Example-20-Adam-Sgdm-with-Tree-Aggregate.ipynb) — Implements distributed optimizers (Adam, SGD with momentum) and uses treeAggregate for efficient gradient reduction. Includes data generation (make_blobs) and plotting utilities; authored notes and reproducible examples are included.

- [Spark-Example-20a-Code-Optimization.ipynb](Spark-Example-20a-Code-Optimization.ipynb) — Practical code-optimization patterns for Spark training loops and data pipelines, with tips for vectorized operations, batching, and memory trade-offs.

- [Spark-Example-20b-Imbalanced-Classes.ipynb](Spark-Example-20b-Imbalanced-Classes.ipynb) — Techniques for handling imbalanced datasets: class weighting, sampling strategies, and evaluation adjustments for skewed labels. Contains data generation for extremely imbalanced scenarios to experiment with weighting.

## Section 8 — MLlib, pipelines and advanced analytics

- [Spark-Example-21-Mllib-Regression.ipynb](Spark-Example-21-Mllib-Regression.ipynb) — Demonstrates building MLlib Pipelines for regression: VectorAssembler, train/test split, LinearRegression model setup, fitting and evaluation. Uses a sample "netflix-subscription" dataset.

- [Spark-Example-22-Mllib-Clustering.ipynb](Spark-Example-22-Mllib-Clustering.ipynb) — KMeans/GMM clustering examples with PCA-based visualization helpers; shows how to prepare features and visualize cluster assignments after dimensionality reduction.

- [Spark-Example-22a-Mllib-Recomender.ipynb](Spark-Example-22a-Mllib-Recomender.ipynb) — ALS recommender example with data ingestion, training and evaluation; explains ALS theory and shows how to prepare and split rating data.

- [Spark-Example-23-Mllib-Sentiment Model.ipynb](Spark-Example-23-Mllib-Sentiment%20Model.ipynb) — End-to-end text classification pipeline for sentiment analysis: preprocessing (HTML removal, regex cleaning), tokenization, TF-IDF, feature selection (Chi-Square) and classifier training with evaluation metrics. Contains sampling and pipeline-building steps to speed up experimentation.

- [Spark-Example-24-Mlib-LDA.ipynb](Spark-Example-24-Mlib-LDA.ipynb) — Topic modeling with LDA on IMDB samples: load/sanitize text, optional lemmatization discussion, vectorization, and topic interpretation. Good for students learning NLP pipelines in Spark.

## Section 9 — NLP, SparkNLP and LLM integrations

- [Spark-Example-25-Gaussian-Mixture-Models-with-EM.ipynb](Spark-Example-25-Gaussian-Mixture-Models-with-EM.ipynb) — Gaussian Mixture Models with EM implemented both from scratch and via library APIs; contains plotting utilities and RDD-based EM examples.

- [Spark-Example-25a-Mllib-Custom-Transformer.ipynb](Spark-Example-25a-Mllib-Custom-Transformer.ipynb) — Shows how to implement custom Transformers for PySpark ML pipelines (custom tokenizer, assembler replacements) and demonstrates how to plug them into a Pipeline.

- [Spark-Example-25b-XGBoost in Spakr.ipynb](Spark-Example-25b-XGBoost in Spakr.ipynb) — Integrates XGBoost with MLlib pipelines on a bike-sharing regression task; includes data description and setup steps for installing xgboost and pyspark when running in notebooks.

- [Spark-Example-26-SparkNLP_Named_Entity_Recognition.ipynb](Spark-Example-26-SparkNLP_Named_Entity_Recognition.ipynb) — Colab-friendly SparkNLP NER pipeline that installs SparkNLP, starts a sparknlp session, and demonstrates pretrained pipelines for entity detection. Run notes: recommends running in Colab or ensuring spark-nlp is installed locally.

- [Spark-Example-27-SparkNLP_Question_Answering_and_Summarization_with_T5.ipynb](Spark-Example-27-SparkNLP_Question_Answering_and_Summarization_with_T5.ipynb) — Demonstrates T5-based QA and summarization using SparkNLP: colab setup, model download notes and pipeline construction. Useful for transformer-style tasks at scale.

- [Spark-Example-28-ChatGPT-Spark-Langchain.ipynb](Spark-Example-28-ChatGPT-Spark-Langchain.ipynb) — Example integration showing how to create a LangChain-style Spark DataFrame agent that uses an LLM (OpenAI) to answer questions over a Spark DataFrame; includes instructions on setting OPENAI_API_KEY and required packages. Run notes: requires network and API keys; be careful with secrets.

- [Spark-Example-29-ChatGPT OpenAI API.ipynb](Spark-Example-29-ChatGPT%20OpenAI%20API.ipynb) — Demonstrates using OpenAI chat completions from a Spark workflow, token-count utilities, batching, and basic prompt construction for tasks like sentiment extraction. Run notes: demonstrates tiktoken usage and how to set API key in a secure file.

- [Spark-Example-30-LLM.ipynb](Spark-Example-30-LLM.ipynb) — Experiments with local and remote LLMs (AutoGGUF, OpenAI, etc.), generation pipelines in Spark and small integration examples showing how to embed LLM inference into distributed workflows.

##  Section 10 — Spark Streaming examples

The `Spark-Example-Steaming/` folder contains a working example of ingesting GDELT-style CSV updates and basic stream-wordcount demos. These files provide both notebook-based demonstrations and runnable Python scripts for local testing.

- `Spark-Streaming.ipynb` — Notebook that walks through Spark Streaming concepts and shows an example stream processing pipeline. It is a teaching-oriented notebook that demonstrates how to set up a streaming source (file/stream), define map/filter/aggregation logic, and inspect micro-batch outputs. Useful for students learning structured or DStream-style streaming concepts depending on the code examples present.

- `gdelt-stream.py` — A script to tail or emit GDELT CSV update files into a directory watched by the streaming job. Helpful to reproduce streaming input locally by replaying timestamped exports.

- `gdelt_update.py` — Helper script to preprocess or normalize GDELT exports (for example, filter columns, fix encodings, or split large export files into micro-batches). Typically used before `gdelt-stream.py` to prepare files for ingestion by the streaming job.

- `stream_wc.py` — A small, focused stream-wordcount script that reads text lines from a streaming source and performs map/flatMap/reduceByKey or equivalent aggregation over windows. Good as the first runnable example to validate a streaming environment.

- `CAMEO.country.txt` — Lookup file (probably mapping CAMEO country codes to names) used to enrich or decode GDELT fields during preprocessing or in the notebook.

- `input_files/` — A set of timestamped GDELT export CSV files that can be used to replay data into a watched directory for the streaming examples. Filenames like `20250612210000.export.CSV` represent time-based slices; use `gdelt-stream.py` to emit them at controlled intervals.

Run notes and recommendations

- Local testing: run `stream_wc.py` or the scripts to verify Python-level streaming logic. To simulate streaming from the provided CSVs, use `gdelt-stream.py` to copy files from `input_files/` into a directory watched by Spark Streaming.

- Cluster testing: ensure the Spark environment supports Structured Streaming or DStreams (matching how the notebook is implemented). If using Structured Streaming, prefer file sources or socket/text sources for local tests.

- Data sensitivity: GDELT data can include many textual fields; be careful with encoding and large fields when replaying. The `gdelt_update.py` script helps normalize files.


# Visual Learning — interactive visualizations and animations

The `Visual-Learning/` folder contains notebooks and animations that help students build intuition about core ML concepts through interactive visuals and animated GIFs.

- [VizLearn-01-Learning-Gradient-Descents-Visualisation.ipynb](Visual-Learning/VizLearn-01-Learning-Gradient-Descents-Visualisation.ipynb) — Visual walkthrough of gradient descent behavior: shows how learning rate, initialization and curvature affect convergence. Contains animated plots and interactive sliders to experiment with step size and visualize trajectories in 2D/3D parameter spaces.

- [VizLearn-02-Overfitting and Regularization.ipynb](Visual-Learning/VizLearn-02-Overfitting%20and%20Regularization.ipynb) — Demonstrates overfitting using polynomial fits and shows L1/L2 regularization effects; includes interactive visualization for model complexity vs validation error.

- [VizLearn-03-Bias-Variance Trade-Off.ipynb](Visual-Learning/VizLearn-03-Bias-Variance%20Trade-Off.ipynb) — Interactive examples illustrating bias vs variance using synthetic data, bootstrapping, and multiple model fits to visualize prediction spread.

- [VizLearn-03b-Bias-Variance Trade-Off.ipynb](Visual-Learning/VizLearn-03b-Bias-Variance%20Trade-Off.ipynb) — Alternate/expanded version of the bias-variance demonstrations with extra visualizations and commentary.

- [VizLearn-04-LogisticRegression_CostFunctionParts_Example.ipynb](Visual-Learning/VizLearn-04-LogisticRegression_CostFunctionParts_Example.ipynb) — Decomposes the logistic regression cost into interpretable parts and visualizes how individual data points affect the loss surface.

- [VizLearn-05-Clustering-K-means.ipynb](Visual-Learning/VizLearn-05-Clustering-K-means.ipynb) — Animated, step-by-step K-means clustering demo showing centroid updates and membership changes; useful for intuition on convergence and initialization effects.

- [VizLearn-06-EM-Method.ipynb](Visual-Learning/VizLearn-06-EM-Method.ipynb) — Visual EM algorithm demo (Gaussian mixture fitting) with component responsibilities and parameter updates plotted per iteration.

- `Animations/` — Contains GIFs used inside the notebooks (k-means animation, etc.). Students can open these directly or view them inline within the notebooks.


