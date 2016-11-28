from __future__ import print_function
import urllib
from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.sql import SQLContext, Row, SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.functions import when
from pyspark.sql.functions import Column
from pyspark.sql.types import StructField, StructType, StringType, IntegerType
import re
import os


class PreProcess(object):
  '''
  This class generates a parquet file with label column and mulitple feature columns.
  The features are processed to remove blank values and replaced with
  - Mean of numerical features
  - Static value for categorical features

  parameters:
  - path_to_input: string
    Path to input data 
  '''

  def __init__(self, raw_input_data):
    self.input_dataframe = raw_input_data


  '''def hash_to_int(self, value):
    if value == "":
      return None
    else:
      x = int(value, 16)
      return x'''


  def blank_as_null(self, x):
    # hash_int = udf(self.hash_to_int, IntegerType())
    return when(x != "", x).otherwise(None)

  def preprocess_data(self):
    '''
    Main pre-processes logic
    '''
    
    changedTypedf = self.input_dataframe.withColumn("I1", self.input_dataframe["I1R"].cast("int")). \
        withColumn("I2", self.input_dataframe["I2R"].cast("int")). \
        withColumn("I3", self.input_dataframe["I3R"].cast("int")). \
        withColumn("I4", self.input_dataframe["I4R"].cast("int")). \
        withColumn("I5", self.input_dataframe["I5R"].cast("int")). \
        withColumn("I6", self.input_dataframe["I6R"].cast("int")). \
        withColumn("I7", self.input_dataframe["I7R"].cast("int")). \
        withColumn("I8", self.input_dataframe["I8R"].cast("int")). \
        withColumn("I9", self.input_dataframe["I9R"].cast("int")). \
        withColumn("I10", self.input_dataframe["I10R"].cast("int")). \
        withColumn("I11", self.input_dataframe["I11R"].cast("int")). \
        withColumn("I12", self.input_dataframe["I12R"].cast("int")). \
        withColumn("I13", self.input_dataframe["I13R"].cast("int"))

    print("Continious features type conversion completed..")
    
    input_filter = changedTypedf.drop("I1R").drop("I2R").drop("I3R").drop("I4R").drop("I5R").drop("I6R"). \
        drop("I7R").drop("I8R").drop("I9R").drop("I10R").drop("I11R").drop("I12R"). \
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

    input_filter_fill = input_filter.fillna({'I1': med_I1[0], 'I2': med_I2[0], 'I3': med_I3[0], 'I4': med_I4[0],
                                             'I5': med_I5[0], 'I6': med_I6[0], 'I7': med_I8[0], 'I8': med_I8[0],
                                             'I9': med_I9[0], 'I10': med_I10[0], 'I11': med_I11[0], 'I12': med_I12[0],
                                             'I13': med_I13[0]})

    print('Missing value handle completed in Continious data..')

    dfWithEmptyReplaced = input_filter_fill.withColumn("C1", self.blank_as_null(input_filter_fill.C1R)).drop("C1R") \
        .withColumn("C2", self.blank_as_null(input_filter_fill.C2R)).drop("C2R") \
        .withColumn("C3", self.blank_as_null(input_filter_fill.C3R)).drop("C3R") \
        .withColumn("C4", self.blank_as_null(input_filter_fill.C4R)).drop("C4R") \
        .withColumn("C5", self.blank_as_null(input_filter_fill.C5R)).drop("C5R") \
        .withColumn("C6", self.blank_as_null(input_filter_fill.C6R)).drop("C6R") \
        .withColumn("C7", self.blank_as_null(input_filter_fill.C7R)).drop("C7R") \
        .withColumn("C8", self.blank_as_null(input_filter_fill.C8R)).drop("C8R") \
        .withColumn("C9", self.blank_as_null(input_filter_fill.C9R)).drop("C9R") \
        .withColumn("C10", self.blank_as_null(input_filter_fill.C10R)).drop("C10R") \
        .withColumn("C11", self.blank_as_null(input_filter_fill.C11R)).drop("C11R") \
        .withColumn("C12", self.blank_as_null(input_filter_fill.C12R)).drop("C12R") \
        .withColumn("C13", self.blank_as_null(input_filter_fill.C13R)).drop("C13R") \
        .withColumn("C14", self.blank_as_null(input_filter_fill.C14R)).drop("C14R") \
        .withColumn("C15", self.blank_as_null(input_filter_fill.C15R)).drop("C15R") \
        .withColumn("C16", self.blank_as_null(input_filter_fill.C16R)).drop("C16R") \
        .withColumn("C17", self.blank_as_null(input_filter_fill.C17R)).drop("C17R") \
        .withColumn("C18", self.blank_as_null(input_filter_fill.C18R)).drop("C18R") \
        .withColumn("C19", self.blank_as_null(input_filter_fill.C19R)).drop("C19R") \
        .withColumn("C20", self.blank_as_null(input_filter_fill.C20R)).drop("C20R") \
        .withColumn("C21", self.blank_as_null(input_filter_fill.C21R)).drop("C21R") \
        .withColumn("C22", self.blank_as_null(input_filter_fill.C22R)).drop("C22R") \
        .withColumn("C23", self.blank_as_null(input_filter_fill.C23R)).drop("C23R") \
        .withColumn("C24", self.blank_as_null(input_filter_fill.C24R)).drop("C24R") \
        .withColumn("C25", self.blank_as_null(input_filter_fill.C25R)).drop("C25R") \
        .withColumn("C26", self.blank_as_null(input_filter_fill.C26R)).drop("C26R")

    print('Categorical features replace empty value..')

    unknown_string_hash = "6e61"
    replace_string = unknown_string_hash

    final_input_data = dfWithEmptyReplaced.fillna({
        "C1": replace_string, "C2": replace_string, "C3": replace_string, "C4": replace_string,
        "C5": replace_string, "C6": replace_string, "C7": replace_string, "C8": replace_string,
        "C9": replace_string, "C10": replace_string, "C11": replace_string, "C12": replace_string,
        "C13": replace_string, "C14": replace_string, "C15": replace_string, "C16": replace_string,
        "C17": replace_string, "C18": replace_string, "C19": replace_string, "C20": replace_string,
        "C21": replace_string, "C22": replace_string, "C23": replace_string, "C24": replace_string,
        "C25": replace_string, "C26": replace_string})

    final_input_data = final_input_data.withColumn("label", final_input_data.label.cast("double"))
    
    return final_input_data
