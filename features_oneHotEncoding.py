from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder,StringIndexer
from pyspark.ml import Pipeline
import sys


spark = SparkSession\
        .builder\
        .getOrCreate()

sc = spark.sparkContext;
sqlContext = SQLContext(sc)
sc.setLogLevel("WARN")

if __name__ == "__main__":

		# Reading the Pre-processed data
        df=spark.read.parquet(sys.argv[1])
       
        col_name=["I1","I2","I3","I4","I5","I6","I7","I8","I9","I10","I11","I12","I13","C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14","C15","C16","C17","C18","C19","C20","C21","C22","C23","C24","C25","C26"]
        
        # Applying String Indexer to all the columns in the dataframe
        stringIndexer = [StringIndexer(inputCol=column, outputCol=column+"Index").fit(df) for column in col_name]
        pipeline = Pipeline(stages=stringIndexer)
        df_indexed = pipeline.fit(df).transform(df)

        # Applying One Hot Encoding to the indexed columns
        encoder = [OneHotEncoder(dropLast=False, inputCol=column+"Index", outputCol=column+"V") for column in col_name]
        pipeline = Pipeline(stages=encoder)
        df_OneHotEncode = pipeline.fit(df_indexed).transform(df_indexed)

        # Reconstructing the dataframe by dropping unnecessary columns
        dropList=["I1","I2","I3","I4","I5","I6","I7","I8","I9","I10","I11","I12","I13","C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14","C15","C16","C17","C18","C19","C20","C21","C22","C23","C24","C25","C26",
                  "I1Index","I2Index","I3Index","I4Index","I5Index","I6Index","I7Index","I8Index","I9Index","I10Index","I11Index","I12Index","I13Index","C1Index","C2Index","C3Index","C4Index","C5Index","C6Index","C7Index","C8Index",
                "C9Index","C10Index","C11Index","C12Index","C13Index","C14Index","C15Index","C16Index","C17Index","C18Index","C19Index","C20Index","C21Index","C22Index","C23Index","C24Index","C25Index","C26Index"]
        df_encoded=df_OneHotEncode.select([column for column in df_OneHotEncode.columns if column not in dropList])

        # Persisting the One Hor Encoded data in parquet file
        df_encoded.write.parquet(sys.argv[2] + "ohe_data.parquet")
