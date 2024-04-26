import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def clean_data(df):
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))

if __name__ == "__main__":
    print("Starting AWS - Spark Application")

    spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()
    scanner = spark.sparkContext
    scanner.setLogLevel('ERROR')
    scanner._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    input_path = "s3://mydeva/ValidationDataset.csv"
    model_path = "s3://mydeva/finalmodel"
 
    df = (spark.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema", 'true')
          .load(input_path))
    df_clean = clean_data(df)
    mdl = PipelineModel.load(model_path)
    predictions = mdl.transform(df_clean)
 
    res = predictions.select(['prediction', 'label'])
    evl = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
    accuracy = evl.evaluate(predictions)
    print(f'Test Accuracy of wine prediction model = {accuracy}')

    # F1 score computation using RDD API
    metrics = MulticlassMetrics(res.rdd.map(tuple))
    f1_score = metrics.weightedFMeasure()
    print(f' F1 Score of Wine prediction = {f1_score}')

    print("Exit")
    spark.stop()
