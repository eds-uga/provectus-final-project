from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.classification import RandomForestClassifier
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

    df = spark.read.parquet(sys.argv[1])

    # df = df.withColumn("label", df.label.cast("double"))
    df_split = df.randomSplit([0.8, 0.2], 9)

    train_df =  df_split[0]
    test_df = df_split[1]

    rfc = RandomForestClassifier(maxDepth=8, maxBins=2400000, numTrees=100)
    rfc_model = rfc.fit(train_df)
    rfc_model.save(sys.argv[2] + "rfc_model")


if __name__ == '__main__':
    main()