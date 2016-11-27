from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
import sys


class AssembleVector(object):
    
    def __init__(self, input_data, output_path=""):
        self.input_data = input_data
        self.output_path = output_path

    def assemble_vector(self):

        write_flag = False

        df = None

        if isinstance(self.input_data, basestring):
            df=spark.read.parquet(self.input_data)
            write_flag = True
        else:
            df = self.input_data

        #Applying Vector Assembler 
        inputColumn = ["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14","C15","C16","C17","C18","C19","C20","C21","C22","C23","C24","C25","C26"]

        stringIndexer = [StringIndexer(inputCol=column, outputCol=column + "I") for column in inputColumn]

        pipeline = Pipeline(stages=stringIndexer)
        df_indexed = pipeline.fit(df).transform(df)

        for column in df_indexed.columns:
            if column in inputColumn:
                df_indexed = df_indexed.drop(column)
        
        column_names = ["I1","I2","I3","I4","I5","I6","I7","I8","I9","I10","I11","I12","I13","C1I","C2I","C3I","C4I","C5I","C6I","C7I","C8I","C9I","C10I","C11I","C12I","C13I","C14I","C15I","C16I","C17I","C18I","C19I","C20I","C21I","C22I","C23I","C24I","C25I","C26I"]

        assembler = VectorAssembler(inputCols=column_names, outputCol="features")
        output = assembler.transform(df_indexed)

        for column in output.columns:
            if column in column_names:
                output = output.drop(column)

        if write_flag:
            #Persisting Assembled Data
            output.write.parquet(self.output_path + "assembled_data.parquet")
        else:
            return output
