from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
import sys


spark = SparkSession\
        .builder\
        .master("local")\
        .appName("Test")\
        .getOrCreate()

sc = spark.sparkContext;
sc.setLogLevel("INFO")


def main():
    df = spark.read.parquet(sys.argv[1])

    df = df.withColumn("label", df.label.cast("double"))
    df_split = df.randomSplit([0.8, 0.2], 9)

    train_df =  df_split[0]
    test_df = df_split[1]

    rfc = RandomForestClassifier(maxDepth=5, maxBins=32, numTrees=5, seed=9)
    rfc_model = rfc.fit(train_df)
    rfc_model.save(sys.argv[2] + "rfc_model")


if __name__ == '__main__':
    main()