import pandas as pd
import os
import tensorflow as tf


def process(df):




    # reading dataset file in a chunk


    # print(df)
    print("****************************BATCH****************")
    # Filling all the missing values in continuous coloumns by the median of respective column
    df['I1'].fillna((df['I1'].median()), inplace=True)
    df['I2'].fillna((df['I2'].median()), inplace=True)
    df['I3'].fillna((df['I3'].median()), inplace=True)
    df['I4'].fillna((df['I4'].median()), inplace=True)
    df['I5'].fillna((df['I5'].median()), inplace=True)
    df['I6'].fillna((df['I6'].median()), inplace=True)
    df['I7'].fillna((df['I7'].median()), inplace=True)
    df['I8'].fillna((df['I8'].median()), inplace=True)
    df['I9'].fillna((df['I9'].median()), inplace=True)
    df['I10'].fillna((df['I10'].median()), inplace=True)
    df['I11'].fillna((df['I11'].median()), inplace=True)
    df['I12'].fillna((df['I12'].median()), inplace=True)
    df['I13'].fillna((df['I13'].median()), inplace=True)

    # Filling all the missing values in categorical coloumns by the mode of respective column
    df['C1'].fillna((df['C1'].mode()[0]), inplace=True)
    df['C2'].fillna((df['C2'].mode()[0]), inplace=True)
    df['C3'].fillna((df['C3'].mode()[0]), inplace=True)
    df['C4'].fillna((df['C4'].mode()[0]), inplace=True)
    df['C5'].fillna((df['C5'].mode()[0]), inplace=True)
    df['C6'].fillna((df['C6'].mode()[0]), inplace=True)
    df['C7'].fillna((df['C7'].mode()[0]), inplace=True)
    df['C8'].fillna((df['C8'].mode()[0]), inplace=True)
    df['C9'].fillna((df['C9'].mode()[0]), inplace=True)
    df['C10'].fillna((df['C10'].mode()[0]), inplace=True)
    df['C11'].fillna((df['C11'].mode()[0]), inplace=True)
    df['C12'].fillna((df['C12'].mode()[0]), inplace=True)
    df['C13'].fillna((df['C13'].mode()[0]), inplace=True)
    df['C14'].fillna((df['C14'].mode()[0]), inplace=True)
    df['C15'].fillna((df['C15'].mode()[0]), inplace=True)
    df['C16'].fillna((df['C16'].mode()[0]), inplace=True)
    df['C17'].fillna((df['C17'].mode()[0]), inplace=True)
    df['C18'].fillna((df['C18'].mode()[0]), inplace=True)
    df['C19'].fillna((df['C19'].mode()[0]), inplace=True)
    df['C20'].fillna((df['C20'].mode()[0]), inplace=True)
    df['C21'].fillna((df['C21'].mode()[0]), inplace=True)
    df['C22'].fillna((df['C22'].mode()[0]), inplace=True)
    df['C23'].fillna((df['C23'].mode()[0]), inplace=True)
    df['C24'].fillna((df['C24'].mode()[0]), inplace=True)
    df['C25'].fillna((df['C25'].mode()[0]), inplace=True)
    df['C26'].fillna((df['C26'].mode()[0]), inplace=True)

    df.to_csv(os.path.join("/home/yash", "Data", "new_file_" + "test.csv"), mode='a',header=False)

#
def main():

    chunksize = 10000

    # name of all the coloumns
    inputColumn = ["Label", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13", "C1",
                   "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
                   "C15", "C16", "C17", "C18", "C19",
                   "C20", "C21", "C22", "C23", "C24", "C25", "C26"]

    for chunk in pd.read_csv("dac_sample.txt", sep='\t', iterator=True, chunksize=chunksize, header=None, names=inputColumn):
        process(chunk)


if __name__ == '__main__':
    main()
