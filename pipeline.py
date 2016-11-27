from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from features_VectorAssembler import AssembleVector
from pre_processing import PreProcess
import sys


spark = SparkSession\
    .builder\
    .master("local")\
    .appName("pipeline")\
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")


def prep_rf(data_input):
    
    # Reference for this piece of code: https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier
    
    # Combine multiple feature columns to one feature column
    av = AssembleVector(data_input)
    data = av.assemble_vector()

    return data


def main():

    # Logistic regression or Random forest
    output_type = sys.argv[1]
    
    raw_file_path = sys.argv[2]

    path_to_output = sys.argv[3]

    pp = PreProcess(raw_file_path)

    preprocessed_data = pp.preprocess_data()

    data = None

    if output_type == "rf":
        data = prep_rf(preprocessed_data)

    # write to file the data variable
    data.write.parquet(path_to_output + "final_" + output_type + "_data.parquet")


if __name__ == '__main__':
    main()
