from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import sys

spark = SparkSession\
        .builder\
        .getOrCreate()



if __name__ == "__main__":

        # Reading the One hot encoded data
        df=spark.read.parquet(sys.argv[1])

        #Applying Vector Assembler 
        inputColumn= ["I1V","I2V","I3V","I4V","I5V","I6V","I7V","I8V","I9V","I10V","I11V","I12V","I13V","C1V","C2V","C3V","C4V","C5V","C6V","C7V","C8V","C9V","C10V","C11V","C12V","C13V","C14V","C15V","C16V","C17V","C18V","C19V",
                        "C20V","C21V","C22V","C23V","C24V","C25V","C26V"]
        assembler = VectorAssembler(inputCols=inputColumn,outputCol="features")
        output = assembler.transform(df)

        # Reconstructing the dataframe by dropping unnecessary columns
        assembledData=output.select([column for column in output.columns if column not in inputColumn])
       
        # Persisting Assembled Data 
        assembledData.write.parquet(sys.argv[2]  + "assembled_data.parquet")

        