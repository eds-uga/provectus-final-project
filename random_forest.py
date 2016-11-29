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

    train_df = spark.read.parquet(sys.argv[1])

    rfc = RandomForestClassifier(maxDepth=8, maxBins=2400000, numTrees=128,impurity="gini")
    rfc_model = rfc.fit(train_df)
    rfc_model.save(sys.argv[2] + "rfc_model")


if __name__ == '__main__':
    main()
