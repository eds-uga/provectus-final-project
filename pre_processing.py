from __future__ import print_function
import urllib
from pyspark import SparkContext, SparkConf,StorageLevel
from pyspark.sql import SQLContext, Row, SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql.functions import  when
from pyspark.sql.functions import Column
import re
import os

spark = SparkSession \
    .builder \
    .appName("CTR") \
    .getOrCreate()
sc = spark.sparkContext;
sqlContext = SQLContext(sc)

def replacetab(line):
    data= line.split('\t')
    return data


def hashtoint(coloumnName):
    x = int(coloumnName, 16)
    return x


def blank_as_null(x):
    return when(x != "", x).otherwise(None)

def main():
    input_data=sc.textFile('gs://dataproc-711992da-9b25-4ec0-b608-4367da5ba1ea-asia/dac_sample.txt',minPartitions=8).map(lambda line: line.encode("utf-8"))
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
              StructField("I13R", StringType(), True),StructField("C1R", StringType(), True),
              StructField("C2R", StringType(), True),
              StructField("C3R", StringType(), True),StructField("C4R", StringType(), True),
              StructField("C5R", StringType(), True),
              StructField("C6R", StringType(), True),StructField("C7R", StringType(), True),
              StructField("C8R", StringType(), True),
              StructField("C9R", StringType(), True),StructField("C10R", StringType(), True),
              StructField("C11R", StringType(), True),
              StructField("C12R", StringType(), True),StructField("C13R", StringType(), True),
              StructField("C14R", StringType(), True),
              StructField("C15R", StringType(), True),StructField("C16R", StringType(), True),
              StructField("C17R", StringType(), True),
              StructField("C18R", StringType(), True),StructField("C19R", StringType(), True),
              StructField("C20R", StringType(), True),
              StructField("C21R", StringType(), True),StructField("C22R", StringType(), True),
              StructField("C23R", StringType(), True),
              StructField("C24R", StringType(), True),StructField("C25R", StringType(), True),
              StructField("C26R", StringType(), True)]

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
    print ("Continious features type conversion completed..")
    input_filter=changedTypedf.drop("I1R").drop("I2R").drop("I3R").drop("I4R").drop("I5R").drop("I6R").\
                                drop("I7R").drop("I8R").drop("I9R").drop("I10R").drop("I11R").drop("I12R").\
                                drop("I13R")
    mean_I1=input_filter.groupBy().avg("I1").head()[0]
    mean_I2 = input_filter.groupBy().avg("I2").head()[0]
    mean_I3 = input_filter.groupBy().avg("I3").head()[0]
    mean_I4 = input_filter.groupBy().avg("I4").head()[0]
    mean_I5 = input_filter.groupBy().avg("I5").head()[0]
    mean_I6 = input_filter.groupBy().avg("I6").head()[0]
    mean_I7 = input_filter.groupBy().avg("I7").head()[0]
    mean_I8 = input_filter.groupBy().avg("I8").head()[0]
    mean_I9 = input_filter.groupBy().avg("I9").head()[0]
    mean_I10 = input_filter.groupBy().avg("I10").head()[0]
    mean_I11 = input_filter.groupBy().avg("I11").head()[0]
    mean_I12 = input_filter.groupBy().avg("I12").head()[0]
    mean_I13 = input_filter.groupBy().avg("I13").head()[0]

    input_filter_fill=input_filter.fillna({'I1':mean_I1,'I2':mean_I2,'I3':mean_I3,'I4':mean_I4,'I5':mean_I5,
                                           'I6': mean_I6,'I7':mean_I8,'I8':mean_I8,'I9':mean_I9,'I10':mean_I10,
                                           'I11': mean_I11,'I12':mean_I12,'I13':mean_I13})
    print ('Missing value handle completed in Continious data..')
    dfWithEmptyReplaced = input_filter_fill.withColumn("C1R", blank_as_null(input_filter_fill.C1R)).\
                                            withColumn("C2R", blank_as_null(input_filter_fill.C2R)).\
                                            withColumn("C3R", blank_as_null(input_filter_fill.C3R)).\
                                            withColumn("C4R", blank_as_null(input_filter_fill.C4R)).\
                                            withColumn("C5R", blank_as_null(input_filter_fill.C5R)).\
                                            withColumn("C6R", blank_as_null(input_filter_fill.C6R)).\
                                            withColumn("C7R", blank_as_null(input_filter_fill.C7R)).\
                                            withColumn("C8R", blank_as_null(input_filter_fill.C8R)).\
                                            withColumn("C9R", blank_as_null(input_filter_fill.C9R)).\
                                            withColumn("C10R", blank_as_null(input_filter_fill.C10R)).\
                                            withColumn("C11R", blank_as_null(input_filter_fill.C11R)).\
                                            withColumn("C12R", blank_as_null(input_filter_fill.C12R)).\
                                            withColumn("C13R", blank_as_null(input_filter_fill.C13R)).\
                                            withColumn("C14R", blank_as_null(input_filter_fill.C14R)).\
                                            withColumn("C15R", blank_as_null(input_filter_fill.C15R)).\
                                            withColumn("C16R", blank_as_null(input_filter_fill.C16R)).\
                                            withColumn("C17R", blank_as_null(input_filter_fill.C17R)).\
                                            withColumn("C18R", blank_as_null(input_filter_fill.C18R)).\
                                            withColumn("C19R", blank_as_null(input_filter_fill.C19R)).\
                                            withColumn("C20R", blank_as_null(input_filter_fill.C20R)).\
                                            withColumn("C21R", blank_as_null(input_filter_fill.C21R)).\
                                            withColumn("C22R", blank_as_null(input_filter_fill.C22R)).\
                                            withColumn("C23R", blank_as_null(input_filter_fill.C23R)).\
                                            withColumn("C24R", blank_as_null(input_filter_fill.C24R)).\
                                            withColumn("C25R", blank_as_null(input_filter_fill.C25R)).\
                                            withColumn("C26R", blank_as_null(input_filter_fill.C26R))
    print ('Categorical features replace empty value..' )
    cnts1 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C1R'].isNotNull()).groupBy("C1R").count().cache()
    str1=cnts1.head()[0]
    cnts2 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C2R'].isNotNull()).groupBy("C2R").count().cache()
    str2 = cnts2.head()[0]
    cnts3 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C3R'].isNotNull()).groupBy("C3R").count().cache()
    str3 = cnts3.head()[0]
    cnts4 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C4R'].isNotNull()).groupBy("C4R").count().cache()
    str4 = cnts4.head()[0]
    cnts5 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C5R'].isNotNull()).groupBy("C5R").count().cache()
    str5 = cnts5.head()[0]
    cnts6 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C6R'].isNotNull()).groupBy("C6R").count().cache()
    str6 = cnts6.head()[0]
    cnts7 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C7R'].isNotNull()).groupBy("C7R").count().cache()
    str7 = cnts7.head()[0]
    cnts8 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C8R'].isNotNull()).groupBy("C8R").count().cache()
    str8 = cnts8.head()[0]
    cnts9 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C9R'].isNotNull()).groupBy("C9R").count().cache()
    str9 = cnts9.head()[0]
    cnts10 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C10R'].isNotNull()).groupBy("C10R").count().cache()
    str10 = cnts10.head()[0]
    cnts11 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C11R'].isNotNull()).groupBy("C11R").count().cache()
    str11 = cnts11.head()[0]
    cnts12 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C12R'].isNotNull()).groupBy("C12R").count().cache()
    str12 = cnts12.head()[0]
    cnts13 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C13R'].isNotNull()).groupBy("C13R").count().cache()
    str13 = cnts13.head()[0]
    cnts14 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C14R'].isNotNull()).groupBy("C14R").count().cache()
    str14 = cnts14.head()[0]
    cnts15 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C15R'].isNotNull()).groupBy("C15R").count().cache()
    str15 = cnts15.head()[0]
    cnts16 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C16R'].isNotNull()).groupBy("C16R").count().cache()
    str16 = cnts16.head()[0]
    cnts17 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C17R'].isNotNull()).groupBy("C17R").count().cache()
    str17 = cnts17.head()[0]
    cnts18 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C18R'].isNotNull()).groupBy("C18R").count().cache()
    str18 = cnts18.head()[0]
    cnts19 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C19R'].isNotNull()).groupBy("C19R").count().cache()
    str19 = cnts19.head()[0]
    cnts20 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C20R'].isNotNull()).groupBy("C20R").count().cache()
    str20 = cnts20.head()[0]
    cnts21 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C21R'].isNotNull()).groupBy("C21R").count().cache()
    str21 = cnts21.head()[0]
    cnts22 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C22R'].isNotNull()).groupBy("C22R").count().cache()
    str22 = cnts22.head()[0]
    cnts23 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C23R'].isNotNull()).groupBy("C23R").count().cache()
    str23 = cnts23.head()[0]
    cnts24 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C24R'].isNotNull()).groupBy("C24R").count().cache()
    str24 = cnts24.head()[0]
    cnts25 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C25R'].isNotNull()).groupBy("C25R").count().cache()
    str25 = cnts25.head()[0]
    cnts26 = dfWithEmptyReplaced.filter(dfWithEmptyReplaced['C26R'].isNotNull()).groupBy("C26R").count().cache()
    str26 = cnts26.head()[0]

    input_data_pre=dfWithEmptyReplaced.fillna({"C1R":str1,"C2R":str2,"C3R":str3,"C4R":str4,"C5R":str5,
                                     "C6R": str6,"C7R":str7,"C8R":str8,"C9R":str9,"C10R":str10,"C11R":str11,
                                     "C12R": str12,"C13R":str13,"C14R":str14,"C15R":str15,"C16R":str16,"C17R":str17,
                                     "C18R": str18,"C19R":str19,"C20R":str20,"C21R":str21,"C22R":str22,"C23R":str23,
                                     "C24R": str24,"C25R":str25,"C26R":str26})
    print ('Replace missing value with mode for Continious features')
    #input_data_pre.show()
    sparkF = udf(hashtoint, IntegerType())

    final_input_data=input_data_pre.select("label","I1","I2","I3","I4","I5","I6","I7","I8","I9","I10","I11","I12","I13",
                                  sparkF("C1R").alias("C1"),sparkF("C2R").alias("C2"),sparkF("C3R").alias("C3"),
                                  sparkF("C4R").alias("C4"),sparkF("C5R").alias("C5"),sparkF("C6R").alias("C6"),
                                  sparkF("C7R").alias("C7"),sparkF("C8R").alias("C8"),sparkF("C9R").alias("C9"),
                                  sparkF("C10R").alias("C10"),sparkF("C11R").alias("C11"),sparkF("C12R").alias("C12"),
                                  sparkF("C13R").alias("C13"),sparkF("C14R").alias("C14"),sparkF("C15R").alias("C15"),
                                  sparkF("C16R").alias("C16"),sparkF("C17R").alias("C17"),sparkF("C18R").alias("C18"),
                                  sparkF("C19R").alias("C19"),sparkF("C20R").alias("C20"),sparkF("C21R").alias("C21"),
                                  sparkF("C22R").alias("C22"),sparkF("C23R").alias("C23"),sparkF("C24R").alias("C24"),
                                  sparkF("C25R").alias("C25"),sparkF("C26R").alias("C26"))
    final_input_data.write.parquet("gs://dataproc-711992da-9b25-4ec0-b608-4367da5ba1ea-asia/data/criteo.parquet")
    print ("Data saved..")

if __name__ == "__main__":
    main()
