from __future__ import print_function
import urllib
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row, SparkSession
from pyspark.sql.types import *
import re
import os

spark = SparkSession \
    .builder \
    .appName("CTR") \
    .getOrCreate()
sc = spark.sparkContext;
sqlContext = SQLContext(sc)


def hashtoint(coloumnName, self):
    for e in coloumnName:
        x = int(e, 16)
        return x

def replacetab(line):
    data= line.split('\t')
    return data

def replaceblank(line):
    input=[]
    for s in line:
        if s is '':
            input.append('Null')
        else:
            input.append(s)
    print (len(input))
    return input

def main():
    input_data=sc.textFile('/home/dharamendra/criteo.txt')\
        .map(lambda line: line.encode("utf-8"))
    process_data=input_data.map(lambda line:replacetab(line))
    fields = [StructField("label", StringType(), True),
              StructField("I1R", StringType(), True),StructField("I2R", StringType(), True),
              StructField("I3R", StringType(), True),
              StructField("I4R", StringType(), True),StructField("I5R", StringType(), True),
              StructField("I6R", StringType(), True),
              StructField("I7R", StringType(), True),StructField("I8R", StringType(), True),
              StructField("I9R", StringType(), True),
              StructField("I10R",StringType(), True),StructField("I11R", StringType(), True),
              StructField("I12R", StringType(), True),
              StructField("I13R", StringType(), True),StructField("C1", StringType(), True),
              StructField("C2", StringType(), True),
              StructField("C3", StringType(), True),StructField("C4", StringType(), True),
              StructField("C5", StringType(), True),
              StructField("C6", StringType(), True),StructField("C7", StringType(), True),
              StructField("C8", StringType(), True),
              StructField("C9", StringType(), True),StructField("C10", StringType(), True),
              StructField("C11", StringType(), True),
              StructField("C12", StringType(), True),StructField("C13", StringType(), True),
              StructField("C14", StringType(), True),
              StructField("C15", StringType(), True),StructField("C16", StringType(), True),
              StructField("C17", StringType(), True),
              StructField("C18", StringType(), True),StructField("C19", StringType(), True),
              StructField("C20", StringType(), True),
              StructField("C21", StringType(), True),StructField("C22", StringType(), True),
              StructField("C23", StringType(), True),
              StructField("C24", StringType(), True),StructField("C25", StringType(), True),
              StructField("C26", StringType(), True)]

    schema = StructType(fields)
    input_dataframe = spark.createDataFrame(process_data, schema)
    changedTypedf = input_dataframe.withColumn("I1", input_dataframe["I1R"].cast(FloatType())).\
                                   withColumn("I2", input_dataframe["I2R"].cast(FloatType())).\
                                   withColumn("I3", input_dataframe["I3R"].cast(FloatType())).\
                                   withColumn("I4", input_dataframe["I4R"].cast(FloatType())).\
                                   withColumn("I5", input_dataframe["I5R"].cast(FloatType())).\
                                   withColumn("I6", input_dataframe["I6R"].cast(FloatType())).\
                                   withColumn("I7", input_dataframe["I7R"].cast(FloatType())).\
                                   withColumn("I8", input_dataframe["I8R"].cast(FloatType())).\
                                   withColumn("I9", input_dataframe["I9R"].cast(FloatType())).\
                                   withColumn("I10", input_dataframe["I10R"].cast(FloatType())).\
                                   withColumn("I11", input_dataframe["I11R"].cast(FloatType())).\
                                   withColumn("I12", input_dataframe["I12R"].cast(FloatType())).\
                                   withColumn("I13", input_dataframe["I13R"].cast(FloatType()))


    #print (schemaTestByte.take(2))
    input_filter=changedTypedf.drop("I1R").drop("I2R").drop("I3R").drop("I4R").drop("I5R").drop("I6R").\
                                drop("I7R").drop("I8R").drop("I9R").drop("I10R").drop("I11R").drop("I12R").\
                                drop("I13R")
    mean_I1=input_filter.groupBy().avg("I1").head()[0]
    print (mean_I1)
    input_filter_fill=input_filter.fillna({'I1':mean_I1})
    input_filter_fill.show()

    sparkF = pyspark.sql.functions.udf(hashtoint, pyspark.sql.types.IntegerType())

    input_filter_fill.select(sparkF(input_filter_fill.workclass)).alias("temp").collect()


if __name__ == "__main__":
    main()
