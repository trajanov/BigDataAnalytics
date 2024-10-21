# How to Run PySpark Jobs on Dataproc Serverless

In this tutorial, you will learn how to run PySpark jobs on Dataproc Serverless, eliminating the need to preprovision a Dataproc cluster.

There are two ways to run Spark jobs on Dataproc Serverless.

1. Using "gcloud" CLI
2. Using the Google Cloud console

# Using gcloud CLI

Due to its “Serverless” nature, Dataproc Serverless does not provision a Spark history server by default, and therefore, it does not store Spark job history. To view the Spark UI for both previous and currently running jobs, you need to configure a single-node persistent Dataproc cluster with a GCS bucket. This cluster is known as a Persistent History Server (PHS). For more detailed information, please refer to this link. https://cloud.google.com/dataproc/docs/concepts/jobs/history-server

## Dataproc Serverless with Persistent History Server (PHS)

This guide explains how to run PySpark jobs on Dataproc Serverless using a Persistent History Server (PHS) cluster.

## Prerequisites

- gcloud CLI is inatlled on your computer. Refer to this link if you need to install gcloud - https://cloud.google.com/sdk/docs/install
- gclous session is initialized. To initialize a gcloud session, run "gcloud auth login" and follow the prompts.
- Ensure that the Google Cloud Storage (GCS) bucket (e.g., `gs://met-cs777-assignment1`) is already created.
- Update the arguments such as `ProjectID`, `Region`, etc., accordingly.
- Dataproc uses a default service account "xxxxxxxxx-compute@developer.gserviceaccount.com" to assume permissions to other resources that it accesses. That service account needs "Storage Object User" role in order to write Spark job history to the GCS bucket you will provide in the commands below. You can provide this permission via the Google Cloud console on the IAM page by editing the service account permissions. 

## Creating a Dataproc Persistent History Server (PHS) Cluster

Dataproc Serverless requires a single-node Dataproc PHS cluster to provision the Spark history server. Without a PHS cluster, Dataproc Serverless does not provide the Spark UI.

### Command to Create PHS Cluster

Substitute the following arguments with values specific to your Google Cloud environment.
- --project
- --region
- --zone
- Cluster name which is provided as "cs777-assignment-1-phs" in this command. It can be any name.
- GCS bucket which is provided as "gs://met-cs777-assignment1" in this command. Rest of the folder structure is created by Dataproc and you don't need to create it.

Execute the command. The output will display the progress, and once the PHS cluster is created, it will terminate. Alternatively, you can monitor the progress on the Google Cloud Console under Dataproc > Cluster.

```sh
gcloud dataproc clusters create cs777-assignment-1-phs \
    --project met-cs-777-434703 \
    --region us-east1 \
    --zone us-east1-b \
    --single-node \
    --master-machine-type n4-standard-2 \
    --master-boot-disk-size 500 \
    --image-version 2.2-debian12 \
    --enable-component-gateway \
    --properties "yarn:yarn.nodemanager.remote-app-log-dir=gs://met-cs777-assignment1/*/,spark:spark.history.fs.logDirectory=gs://met-cs777-assignment1/*/spark-job-historyyarn-logs,spark:spark.eventLog.dir=gs://met-cs777-assignment1/events/spark-job-history"

```

The command will wait at this line "Waiting for cluster creation operation...⠛" until the cluster is created.

### Command to Submit PySpark Jobs to Dataproc Serverless

Now that you have successfully created a PHS cluster, it’s time to submit a PySpark job to Dataproc Serverless. Execute the following command, replacing the arguments with those specific to your Google Cloud environment. Ensure that all prerequisites are met before proceeding.

#### Prerequisites

- Upload your PySpark file into a GCS bucket in your project. It can be in any folder in the GCS bucket.

Substitute the following arguments with values specific to your Google Cloud environment.
- --project
- --region
- --history-server-cluster with the fully qualified name of your PHS cluster. The fully qualified cluster name in the following command is provided as "projects/met-cs-777-434703/regions/us-east1/clusters/cs777-assignment-1-phs" where "met-cs-777-434703" is the project name, "us-east1" is the region, "cs777-assignment-1-phs" is thw PHS cluster name you had given in the previous command.
- PySpark file path in GCS which is provided as "gs://met-cs777-assignment1/Yawale_Pankaj_Assignment_5.py" in this command.
- --properties which takes Spark driver and executor related settings that are desired in your case. You can use this option to override the default setting provided by Dataproc Serverless. If you keep the same settings as in the command below, your Spark job will run with 5 executors with 4 cores each and 4 cores for the driver. Memory is managed by Dataproc Serverless.
- spark.app.name with the name that you would like to give to your Spark job
- Any system arguments that your PySpark code expects by providing those at the end after "-- " (space after two dashes). Multiple arguments should to be separated by a space.

```sh
gcloud dataproc batches submit \
    --project met-cs-777-434703 \
    --region us-east1 \
    pyspark gs://met-cs777-assignment1/Yawale_Pankaj_Assignment_5.py \
    --history-server-cluster=projects/met-cs-777-434703/regions/us-east1/clusters/cs777-assignment-1-phs \
    --properties spark.executor.instances=5,spark.driver.cores=4,spark.executor.cores=4,spark.app.name=Yawale_Pankaj_Assignment_4 \
    -- gs://met-cs-777-data/TrainingData.txt.bz2 gs://met-cs-777-data/TestingData.txt.bz2

```

The command will show the progress of your job. Terminating the command locally, doesn't terminate your Spark job. You will see the outout of your job being written to the command console once the job starts running successfully.

Go to "Dataproc > Serverless > Batches" in the Google Cloud console. Your Spark job will appear as a batch with the initial status as "Pending" which changes to "Running" in a few minutes. If the job encounters any errros, it will transition to the "Failed" status.

Click on the BatchID link, which will take you to the Dataproc Serverless overview page of your Spakr job, including the job output. It will look similar to this screenshot.

![Dataproc Serverless Job Overview](https://raw.githubusercontent.com/kiat/BigDataAnalytics/master/Installations-HowTos/sceenshots/dataproc_serverless_job_overview.jpg "Dataproc Serverless Job Overview")

### Deleting Dataproc Serverless Job
Dataproc Serverless will automatically terminate your Spark job once it's finished and destroy the infrastructure resources that it has provsioned to run it.

But If you need to stop your Dataproc Serverless Job (Batch) prematurely for some reason, you can do so via the Google Cloud console by selecting your Batch and clicking on the delete button. 

![PHS Cluster](https://raw.githubusercontent.com/kiat/BigDataAnalytics/master/Installations-HowTos/sceenshots/dataproc_serverless_delete_batch.jpg "PHS Cluster")


Alternatively, you can use the following command. Replace the BatchID ("36e754a79e224d3286c8dbc941d74153") with the ID of your Dataproc Serverless batch and other arguments with values specific to your Google Cloud project.

```sh
gcloud dataproc batches cancel 36e754a79e224d3286c8dbc941d74153 \
    --project met-cs-777-434703 \
    --region us-east1

```

### Spark History Server UI

At this time, the Spark job progress is being written into the GCS bucket by the PHS cluster. Follow the below steps to see the Spark UI.

- Go to Dataproc > Clusters
- Click on ther PHS cluster
- Click on the "WEB INTERFACES" tab
- Click on the "Spark History Server" link


![PHS Cluster](https://raw.githubusercontent.com/kiat/BigDataAnalytics/master/Installations-HowTos/sceenshots/dataproc_serverless_phs_cluster.jpg "PHS Cluster")

To see your Spark job on the UI, click on the "Show incomplete applications" link and use your prior knowledge of the Spark UI to monitor your job.

![Spark History Server UI](https://raw.githubusercontent.com/kiat/BigDataAnalytics/master/Installations-HowTos/sceenshots/dataproc_serverless_spark_history_ui.jpg "Spark History Server UI")


Congratulations!! You've successfully used Dataproc Serverless to run yout Spark job!

# Clean Up

## Delete PHS Cluster

If you prefer not to keep the PHS cluster running continuously to save costs, you can delete it and recreate it later using the same steps. Deleting and recreating the PHS cluster will not remove the Spark history server state stored in GCS. By recreating the PHS cluster with the same GCS bucket, you will retain the history of all previously run Spark jobs.

Run the following command to delete your PHS cluster. Update the argument values to match your Google Cloud project and the existing PHS cluster name you used when creating it.

```sh
gcloud dataproc clusters delete cs777-assignment-1-phs \
    --project met-cs-777-434703 \
    --region us-east1

```



