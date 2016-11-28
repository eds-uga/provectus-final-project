from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import StructField, StructType, StringType, IntegerType
from features_VectorAssembler import AssembleVector
from pre_processing import PreProcess
import sys


spark = SparkSession \
   .builder \
   .appName("CTR") \
   .config("spark.driver.maxResultSize","3g")\
   .config("spark.executor.heartbeatInterval","60s")\
   .config("spark.storage.memoryMapThreshold","16m")\
   .config("spark.rdd.compress","true")\
   .getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)
sc.setLogLevel("WARN")
# .config('spark.sql.warehouse.dir', 'file:///C:/')\

def prep_rf(data_input):
    
    # Reference for this piece of code: https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier
    
    # Combine multiple feature columns to one feature column
    av = AssembleVector(data_input)
    data = av.assemble_vector()

    return data


def replacetab(line):
    data = line.split('\t')
    return data

def create_df(process_data):
    fields = [StructField("label", StringType(), True),
              StructField("I1R", StringType(), True), StructField("I2R", StringType(), True),
              StructField("I3R", StringType(), True),
              StructField("I4R", StringType(), True), StructField("I5R", StringType(), True),
              StructField("I6R", StringType(), True),
              StructField("I7R", StringType(), True), StructField("I8R", StringType(), True),
              StructField("I9R", StringType(), True),
              StructField("I10R", StringType(), True), StructField("I11R", StringType(), True),
              StructField("I12R", StringType(), True),
              StructField("I13R", StringType(), True), StructField("C1R", StringType(), True),
              StructField("C2R", StringType(), True),
              StructField("C3R", StringType(), True), StructField("C4R", StringType(), True),
              StructField("C5R", StringType(), True),
              StructField("C6R", StringType(), True), StructField("C7R", StringType(), True),
              StructField("C8R", StringType(), True),
              StructField("C9R", StringType(), True), StructField("C10R", StringType(), True),
              StructField("C11R", StringType(), True),
              StructField("C12R", StringType(), True), StructField("C13R", StringType(), True),
              StructField("C14R", StringType(), True),
              StructField("C15R", StringType(), True), StructField("C16R", StringType(), True),
              StructField("C17R", StringType(), True),
              StructField("C18R", StringType(), True), StructField("C19R", StringType(), True),
              StructField("C20R", StringType(), True),
              StructField("C21R", StringType(), True), StructField("C22R", StringType(), True),
              StructField("C23R", StringType(), True),
              StructField("C24R", StringType(), True), StructField("C25R", StringType(), True),
              StructField("C26R", StringType(), True)]

    schema = StructType(fields)

    return spark.createDataFrame(process_data, schema)


def main():

    # Logistic regression or Random forest
    output_type = sys.argv[1]
    
    raw_file_path = sys.argv[2]

    path_to_output = sys.argv[3]

    raw_input_rdd = sc.textFile(raw_file_path, minPartitions=32).map(lambda line: line.encode("utf-8"))

    process_data = raw_input_rdd.map(lambda line: replacetab(line))

    df_for_pp = create_df(process_data)

    pp = PreProcess(df_for_pp)

    preprocessed_data = pp.preprocess_data()

    data = None

    if output_type == "rf":
        data = prep_rf(preprocessed_data)

    # write to file the data variable
    data.write.parquet(path_to_output + "final_" + output_type + "_data.parquet")


if __name__ == '__main__':
    main()
