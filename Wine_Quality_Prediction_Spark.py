import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def clean_data(df):
    """Cleans data by casting columns to double and stripping extra quotes."""
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))

if __name__ == "__main__":
    print("Starting Spark Application")

    spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    # Ensure that the S3 configurations are set correctly
    sc._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    # Paths to your S3 buckets
    input_path = "s3://mydeva/ValidationDataset.csv"
    model_path = "s3://mydeva/finalmodel"

    # Load the validation dataset from S3
    df = (spark.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema", 'true')
          .load(input_path))

    # Clean and prepare the data
    df_clean = clean_data(df)

    # Load the trained model from S3
    model = PipelineModel.load(model_path)

    # Make predictions
    predictions = model.transform(df_clean)

    # Select the necessary columns and compute evaluation metrics
    results = predictions.select(['prediction', 'label'])
    evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)
    print(f'Test Accuracy of wine prediction model = {accuracy}')

    # F1 score computation using RDD API
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    f1_score = metrics.weightedFMeasure()
    print(f'Weighted F1 Score of wine prediction model = {f1_score}')

    print("Exiting Spark Application")
    spark.stop()
