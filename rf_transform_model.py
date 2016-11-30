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

    df_test, df_discard = df_test.randomSplit([0.2, 0.8])

    # Load the model
    rf_model=RandomForestClassificationModel.load(sys.argv[2])
    
    # Make the predictions
    predictions = rf_model.transform(df_test)
    
    #predictionsRDD=predictions.rdd

    #predictionsRDD.saveAsTextFile(sys.argv[3]+"output.text")

    evaluator_acc = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="accuracy")
    accuracy = evaluator_acc.evaluate(predictions)

    print "accuracy *******************"
    print accuracy

    evaluator_pre = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="weightedPrecision")
    
    print "precision *******************"
    print evaluator_pre.evaluate(predictions)

    print "recall **********************"
    print MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="weightedRecall").evaluate(predictions)


if __name__ == '__main__':
    main()