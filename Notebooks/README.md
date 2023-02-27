# Learning PySpark by examples

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

# PySpark by examples

### RDD Basics

* [Spark-Example-01-Spark-Word-Count-Example.ipynb](Spark-Example-01-Spark-Word-Count-Example.ipynb): This Jupyter notebook provides a basic example of using Apache Spark to count the occurrences of words in a text file.
* [Spark-Example-02-RDD Basics.ipynb](Spark-Example-02-RDD%20Basics.ipynb): This Jupyter notebook covers the basics of working with Resilient Distributed Datasets (RDDs) in Apache Spark.
* [Spark-Example-03-RDD transformations.ipynb](Spark-Example-03-RDD%20transformations.ipynb): This Jupyter notebook contains examples of various transformations that can be applied to RDDs in Apache Spark.
* [Spark-Example-04-Not to use transformations.ipynb](Spark-Example-04-Not%20to%20use%20transformations.ipynb): This Jupyter notebook explains some transformations in Apache Spark that can be computationally expensive and should be avoided.
* [Spark-Example-05-Flight-dataset.ipynb](Spark-Example-05-Flight-dataset.ipynb): This Jupyter notebook provides an example of using Apache Spark to analyze flight data.
* [Spark-Example-06-Join Operation on RDD.ipynb](Spark-Example-06-Join%20Operation%20on%20RDD.ipynb): This Jupyter notebook provides examples of how to perform join operations on RDDs in Apache Spark.
* [Spark-Example-07-RDD-Dataframe-Examples-Operations.ipynb](Spark-Example-07-RDD-Dataframe-Examples-Operations.ipynb): This Jupyter notebook provides examples of how to work with RDDs and DataFrames in Apache Spark.

### RDD Advanced

- [Spark-Example-08-Data-Partitioning.ipynb](Spark-Example-08-Data-Partitioning.ipynb): This Jupyter notebook explains the concept of data partitioning in Apache Spark and how it can improve performance.
- [Spark-Example-09-Data-Partitioning-TreeAggregate.ipynb](Spark-Example-09-Data-Partitioning-TreeAggregate.ipynb): This Jupyter notebook provides an example of using tree aggregation for efficient computation on a large dataset in Apache Spark.
- [Spark-Example-09.1-Data-MapPartition.ipynb](Spark-Example-09.1-Data-MapPartition.ipynb): This Jupyter notebook explains the mapPartition transformation in Apache Spark and provides examples of how to use it.
- [Spark-Example-09.2-treeAggregate-Min-Max-Calculation.ipynb)](Spark-Example-09.2-treeAggregate-Min-Max-Calculation.ipynb)): This Jupyter notebook provides an example of using tree aggregation in Apache Spark to efficiently calculate the minimum and maximum values in a large dataset.
- [Spark-Example-09.3-PySpark and NumPy.ipynb](Spark-Example-09.3-PySpark%20and%20NumPy.ipynb): This Jupyter notebook contains examples of how to use NumPy arrays with PySpark in Apache Spark.
- [Spark-Example-13-AggregateByKey.ipynb](Spark-Example-13-AggregateByKey.ipynb): This Jupyter notebook provides examples of how to use the aggregateByKey transformation in Apache Spark.
- [Spark-Example-14-combinedByKey.ipynb](Spark-Example-14-combinedByKey.ipynb): This Jupyter notebook provides examples of how to use the combineByKey transformation in Apache Spark.
- [Spark-Example-15-forloop-Variable.ipynb](Spark-Example-15-forloop-Variable.ipynb): This Jupyter notebook provides examples of how to use the for loop variable in Apache Spark.
- [Spark-Example-16-Map-Side-Join.ipynb](Spark-Example-16-Map-Side-Join.ipynb): This Jupyter notebook provides an example of using map-side join in Apache Spark.

# PySpark DataFrames

- [Spark-Example-10-DataFrames.ipynb](Spark-Example-10-DataFrames.ipynb): This Jupyter notebook introduces DataFrames in Apache Spark and provides examples of how to work with them.
- [Spark-Example-10-Working-With-Dataframes-Array.ipynb](Spark-Example-10-Working-With-Dataframes-Array.ipynb): This Jupyter notebook provides examples of how to work with arrays in DataFrames in Apache Spark.
- [Spark-Example-11-Working-With-Dataframes-Array-3.ipynb](Spark-Example-11-Working-With-Dataframes-Array-3.ipynb): This Jupyter notebook provides examples of how to work with arrays in DataFrames in Apache Spark, including creating arrays and performing operations on them.
- [Spark-Example-12-Working-With-Dataframes.ipynb](Spark-Example-12-Working-With-Dataframes.ipynb): This Jupyter notebook covers various operations that can be performed on DataFrames in Apache Spark.

# ML models implementation

- [Spark-Example-17-Simple-Linear-Regression-Implementation.ipynb](Spark-Example-17-Simple-Linear-Regression-Implementation.ipynb): This Jupyter notebook provides an example of implementing a simple linear regression in Apache Spark.
- [Spark-Example-18-Linear-Regression-Gradient Descent.ipynb](Spark-Example-18-Linear-Regression-Gradient%20Descent.ipynb): This Jupyter notebook provides an example of implementing linear regression with gradient descent in Apache Spark.
- [Spark-Example-19-Logistic-Regression.ipynb](Spark-Example-19-Logistic-Regression.ipynb): This Jupyter notebook provides an example of implementing logistic regression in Apache Spark.
- [Spark-Example-20-Adam-Sgdm-with-Tree-Aggregate.ipynb](Spark-Example-20-Adam-Sgdm-with-Tree-Aggregate.ipynb): This Jupyter notebook provides an example of using Adam and SGDM optimization algorithms with tree aggregation in Apache Spark.
- [Spark-Example-20a-Code-Optimization.ipynb](Spark-Example-20a-Code-Optimization.ipynb): This Jupyter notebook provides examples of code optimization of the Spark-Example-20-Adam-Sgdm-with-Tree-Aggregate.ipynb Notebook
- [Spark-Example-20b-Imbalanced-Classes.ipynb](Spark-Example-20b-Imbalanced-Classes.ipynb): This Jupyter notebook provides an example of how to handle imbalanced classes in machine learning using Apache Spark.
- [Spark-Example-25-Gaussian-Mixture-Models-with-EM.ipynb](Spark-Example-25-Gaussian-Mixture-Models-with-EM.ipynb): This Jupyter notebook provides an example of using Gaussian mixture models with the expectation-maximization (EM) algorithm in Apache Spark.

# Spark MLlib examples

- [Spark-Example-21-Mllib-Regression copy.ipynb](Spark-Example-21-Mllib-Regression%20copy.ipynb): This Jupyter notebook provides examples of regression algorithms in Apache Spark's MLlib library.
- [Spark-Example-22-Mllib-Clustering.ipynb](Spark-Example-22-Mllib-Clustering.ipynb): This Jupyter notebook provides examples of clustering algorithms in Apache Spark's MLlib library.
- [Spark-Example-23-Mllib-Sentiment Model.ipynb](Spark-Example-23-Mllib-Sentiment%20Model.ipynb): This Jupyter notebook provides an example of building a sentiment analysis model in Apache Spark's MLlib library.
- [Spark-Example-24-Mlib-LDA.ipynb](Spark-Example-24-Mlib-LDA.ipynb): This Jupyter notebook provides an example of using Latent Dirichlet Allocation (LDA) for topic modeling in Apache Spark's MLlib library.

### SparkNLP Examples

- [Spark-Example-26-SparkNLP_Named_Entity_Recognition.ipynb](Spark-Example-26-SparkNLP_Named_Entity_Recognition.ipynb): This Jupyter notebook provides an example of using Spark NLP for named entity recognition in text data.
- [Spark-Example-27-SparkNLP_Question_Answering_and_Summarization_with_T5.ipynb](Spark-Example-27-SparkNLP_Question_Answering_and_Summarization_with_T5.ipynb): This Jupyter notebook provides an example of using Spark NLP with the T5 transformer for question answering and summarization tasks in natural language processing.

# Python ML-related examples

- [Python-Example-Gradient-Descents-Visualisation.ipynb](Python-Example-Gradient-Descents-Visualisation.ipynb): This Jupyter notebook provides examples and visualizations of gradient descent algorithms in Python.
- [Python-LogisticRegression_CostFunctionParts_Example.ipynb](Python-LogisticRegression_CostFunctionParts_Example.ipynb): This Jupyter notebook contains an example implementation of logistic regression in Python, along with an explanation of the different parts of the cost function.
- [Python-Overfitting and Regularization.ipynb](Python-Overfitting%20and%20Regularization.ipynb): This Jupyter notebook covers the concepts of overfitting and regularization in machine learning, with examples and visualizations in Python.
