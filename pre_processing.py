from __future__ import print_function
import urllib
from pyspark import SparkContext, SparkConf,StorageLevel
from pyspark.sql import SQLContext, Row, SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.functions import  when
from pyspark.sql.functions import Column
from pyspark.sql.types import StructField, StructType, StringType
import re
import os
import sys


spark = SparkSession \
    .builder \
    .appName("CTR") \
    .getOrCreate()
sc = spark.sparkContext;
sqlContext = SQLContext(sc)


def replacetab(line):
    data= line.split('\t')
    return data


def blank_as_null(x):
    return when(x != "", x).otherwise(None)


def main():
    input_data=sc.textFile(sys.argv[1], minPartitions=8).map(lambda line: line.encode("utf-8"))
    input_data.persist(StorageLevel(True, False, False, False, 1))
    print ("Input Data read completed...")
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
    changedTypedf = input_dataframe.withColumn("I1", input_dataframe["I1R"].cast("int")).\
                                   withColumn("I2", input_dataframe["I2R"].cast("int")).\
                                   withColumn("I3", input_dataframe["I3R"].cast("int")).\
                                   withColumn("I4", input_dataframe["I4R"].cast("int")).\
                                   withColumn("I5", input_dataframe["I5R"].cast("int")).\
                                   withColumn("I6", input_dataframe["I6R"].cast("int")).\
                                   withColumn("I7", input_dataframe["I7R"].cast("int")).\
                                   withColumn("I8", input_dataframe["I8R"].cast("int")).\
                                   withColumn("I9", input_dataframe["I9R"].cast("int")).\
                                   withColumn("I10", input_dataframe["I10R"].cast("int")).\
                                   withColumn("I11", input_dataframe["I11R"].cast("int")).\
                                   withColumn("I12", input_dataframe["I12R"].cast("int")).\
                                   withColumn("I13", input_dataframe["I13R"].cast("int"))

    print ("Continious features type conversion completed..")
    input_filter=changedTypedf.drop("I1R").drop("I2R").drop("I3R").drop("I4R").drop("I5R").drop("I6R").\
                                drop("I7R").drop("I8R").drop("I9R").drop("I10R").drop("I11R").drop("I12R").\
                                drop("I13R")

    med_I1 = input_filter.filter(input_filter['I1'].isNotNull()).approxQuantile("I1", [0.5], 0.25)
    med_I2 = input_filter.filter(input_filter['I2'].isNotNull()).approxQuantile("I2", [0.5], 0.25)
    med_I3 = input_filter.filter(input_filter['I3'].isNotNull()).approxQuantile("I3", [0.5], 0.25)
    med_I4 = input_filter.filter(input_filter['I4'].isNotNull()).approxQuantile("I4", [0.5], 0.25)
    med_I5 = input_filter.filter(input_filter['I5'].isNotNull()).approxQuantile("I5", [0.5], 0.25)
    med_I6 = input_filter.filter(input_filter['I6'].isNotNull()).approxQuantile("I6", [0.5], 0.25)
    med_I7 = input_filter.filter(input_filter['I7'].isNotNull()).approxQuantile("I7", [0.5], 0.25)
    med_I8 = input_filter.filter(input_filter['I8'].isNotNull()).approxQuantile("I8", [0.5], 0.25)
    med_I9 = input_filter.filter(input_filter['I9'].isNotNull()).approxQuantile("I9", [0.5], 0.25)
    med_I10 = input_filter.filter(input_filter['I10'].isNotNull()).approxQuantile("I10", [0.5], 0.25)
    med_I11 = input_filter.filter(input_filter['I11'].isNotNull()).approxQuantile("I11", [0.5], 0.25)
    med_I12 = input_filter.filter(input_filter['I12'].isNotNull()).approxQuantile("I12", [0.5], 0.25)
    med_I13 = input_filter.filter(input_filter['I13'].isNotNull()).approxQuantile("I13", [0.5], 0.25)

    input_filter_fill=input_filter.fillna({'I1': med_I1[0],'I2': med_I2[0],'I3': med_I3[0],'I4': med_I4[0],\
      'I5': med_I5[0],'I6': med_I6[0],'I7': med_I8[0],'I8': med_I8[0],'I9': med_I9[0],'I10': med_I10[0],'I11': med_I11[0],'I12': med_I12[0],'I13': med_I13[0]})

    print ('Missing value handle completed in Continious data..')

    dfWithEmptyReplaced = input_filter_fill.withColumn("C1", blank_as_null(input_filter_fill.C1)).\
                                            withColumn("C2", blank_as_null(input_filter_fill.C2)).\
                                            withColumn("C3", blank_as_null(input_filter_fill.C3)).\
                                            withColumn("C4", blank_as_null(input_filter_fill.C4)).\
                                            withColumn("C5", blank_as_null(input_filter_fill.C5)).\
                                            withColumn("C6", blank_as_null(input_filter_fill.C6)).\
                                            withColumn("C7", blank_as_null(input_filter_fill.C7)).\
                                            withColumn("C8", blank_as_null(input_filter_fill.C8)).\
                                            withColumn("C9", blank_as_null(input_filter_fill.C9)).\
                                            withColumn("C10", blank_as_null(input_filter_fill.C10)).\
                                            withColumn("C11", blank_as_null(input_filter_fill.C11)).\
                                            withColumn("C12", blank_as_null(input_filter_fill.C12)).\
                                            withColumn("C13", blank_as_null(input_filter_fill.C13)).\
                                            withColumn("C14", blank_as_null(input_filter_fill.C14)).\
                                            withColumn("C15", blank_as_null(input_filter_fill.C15)).\
                                            withColumn("C16", blank_as_null(input_filter_fill.C16)).\
                                            withColumn("C17", blank_as_null(input_filter_fill.C17)).\
                                            withColumn("C18", blank_as_null(input_filter_fill.C18)).\
                                            withColumn("C19", blank_as_null(input_filter_fill.C19)).\
                                            withColumn("C20", blank_as_null(input_filter_fill.C20)).\
                                            withColumn("C21", blank_as_null(input_filter_fill.C21)).\
                                            withColumn("C22", blank_as_null(input_filter_fill.C22)).\
                                            withColumn("C23", blank_as_null(input_filter_fill.C23)).\
                                            withColumn("C24", blank_as_null(input_filter_fill.C24)).\
                                            withColumn("C25", blank_as_null(input_filter_fill.C25)).\
                                            withColumn("C26", blank_as_null(input_filter_fill.C26))

    print ('Categorical features replace empty value..' )

    replace_string = "unknown"

    final_input_data = dfWithEmptyReplaced.fillna({
      "C1": replace_string, "C2": replace_string, "C3" : replace_string, "C4" : replace_string,
      "C5": replace_string, "C6": replace_string, "C7" : replace_string, "C8" : replace_string,
      "C9": replace_string, "C10": replace_string, "C11" : replace_string, "C12" : replace_string,
      "C13": replace_string, "C14": replace_string, "C15": replace_string,"C16": replace_string,
      "C17": replace_string, "C18": replace_string, "C19": replace_string, "C20": replace_string,
      "C21": replace_string, "C22": replace_string, "C23": replace_string, "C24": replace_string,
      "C25": replace_string,"C26":replace_string})

    print ('Replace missing value with mode for Continious features')

    final_input_data.write.parquet(sys.argv[2] + "data_with_medians.parquet")
    print ("Data saved..")


if __name__ == "__main__":
    main()
