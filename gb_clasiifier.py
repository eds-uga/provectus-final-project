from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
import sys


def main():
    spark = SparkSession \
        .builder \
        .appName("RandomForest") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .getOrCreate()

    sc = spark.sparkContext
    sqlContext = SQLContext(sc)

    sc.setLogLevel("INFO")

    # Loading the test data
    df_test = spark.read.parquet(sys.argv[1])

    df_test, df_train = df_test.randomSplit([0.3, 0.7])
    df_train_indexed=df_train.selectExpr("label as indexedLabel","features as indexedFeatures")
    df_test_indexed=df_test.selectExpr("label as indexedLabel","features as indexedFeatures")

    # # Load the model
    # rf_model = RandomForestClassificationModel.load(sys.argv[2])
    #
    # # Make the predictions
    # predictions = rf_model.transform(df_test)
    gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=100,maxBins=24000000)
    model=gbt.fit(df_train_indexed)
    predictions = model.transform(df_test_indexed)

    # predictionsRDD=predictions.rdd

    # predictionsRDD.saveAsTextFile(sys.argv[3]+"output.text")

    evaluator_acc = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="indexedLabel",
                                                      metricName="accuracy")
    accuracy = evaluator_acc.evaluate(predictions)

    print "accuracy *******************"
    print accuracy

    evaluator_pre = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="indexedLabel",
                                                      metricName="weightedPrecision")

    print "precision *******************"
    print evaluator_pre.evaluate(predictions)

    print "recall **********************"
    print MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="indexedLabel",
                                            metricName="weightedRecall").evaluate(predictions)


if __name__ == '__main__':
    main()