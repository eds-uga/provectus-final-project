from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.feature import StringIndexer, VectorAssembler, ChiSqSelector
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

        # columns for StringIndexer
        inputColumn = ["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14","C15","C16","C17","C18","C19","C20","C21","C22","C23","C24","C25","C26"]

        stringIndexer = [StringIndexer(inputCol=column, outputCol=column + "I") for column in inputColumn]

        pipeline = Pipeline(stages=stringIndexer)
        df_indexed = pipeline.fit(df).transform(df)

        df_indexed = df_indexed.drop("C1").drop("C2").drop("C3").drop("C4").drop("C5").drop("C6")\
            .drop("C7").drop("C8").drop("C9").drop("C10").drop("C11")\
            .drop("C12").drop("C13").drop("C14").drop("C15").drop("C16").drop("C17").drop("C18").drop("C19")\
            .drop("C20").drop("C21").drop("C22").drop("C23").drop("C24").drop("C25").drop("C26")

        # Combine categorical features to feed into chi squared selector
        cat_column_names = ["C1I","C2I","C5I","C6I","C8I","C9I","C11I","C13I","C14I",\
        "C17I","C18I","C19I","C20I","C22I","C23I","C25I"]

        cat_assembler = VectorAssembler(inputCols=cat_column_names, outputCol="cat_features")
        cat_output = cat_assembler.transform(df_indexed)
        # combine complete

        # Applying chi-squared selector
        chiSq = ChiSqSelector(numTopFeatures=13, featuresCol="cat_features", outputCol="selectedFeatures", labelCol="label")
        chi_result = chiSq.fit(cat_output).transform(cat_output)
        # chi result obtained
        
        # columns to assemble in final output
        column_names = ["I1","I2","I3","I4","I5","I6","I7","I8","I9","I10","I11","I12","I13","selectedFeatures"]

        assembler = VectorAssembler(inputCols=column_names, outputCol="features")
        output = assembler.transform(chi_result)
        # combining complete

        # drop extraneous columns
        output = output.drop("I1").drop("I2").drop("I3").drop("I4").drop("I5").drop("I6").drop("I7")\
        .drop("I8").drop("I9").drop("I10").drop("I11").drop("I12").drop("I13").drop("C1I").drop("C2I")\
        .drop("C3I").drop("C4I").drop("C5I").drop("C6I").drop("C7I").drop("C8I").drop("C9I").drop("C10I")\
        .drop("C11I").drop("C12I").drop("C13I").drop("C14I").drop("C15I").drop("C16I").drop("C17I")\
        .drop("C18I").drop("C19I").drop("C20I").drop("C21I").drop("C22I").drop("C23I").drop("C24I")\
        .drop("C25I").drop("C26I").drop("cat_features").drop("selectedFeatures")

        if write_flag:
            #Persisting Assembled Data
            output.write.parquet(self.output_path + "assembled_data.parquet")
        else:
            return output
