from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys


def main():
    spark = SparkSession \
       .builder \
       .appName("RandomForest") \
       .config("spark.executor.heartbeatInterval","60s")\
       .getOrCreate()

    sc = spark.sparkContext
    sqlContext = SQLContext(sc)
    
    sc.setLogLevel("INFO")

    # Loading the test data
    df_test= spark.read.parquet(sys.argv[1])

    # Load the model
    rf_model=RandomForestClassificationModel.load(sys.argv[2])
    
    # Make the predictions
    predictions = rf_model.transform(df_test)
    
    predictionsRDD=predictions.rdd

    predictionsRDD.saveAsTextFile(sys.argv[3]+"output.text")


if __name__ == '__main__':
    main()