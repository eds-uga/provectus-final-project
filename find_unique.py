import pandas as pd

set_C1 = set()
set_C2 = set()
set_C3 = set()
set_C4 = set()
set_C5 = set()
set_C6 = set()
set_C7 = set()
set_C8 = set()
set_C9 = set()
set_C10 = set()
set_C11 = set()
set_C12 = set()
set_C13 = set()
set_C14 = set()
set_C15 = set()
set_C16 = set()
set_C17 = set()
set_C18 = set()
set_C19 = set()
set_C20 = set()
set_C21 = set()
set_C22 = set()
set_C23 = set()
set_C24 = set()
set_C25 = set()
set_C26 = set()

def finduniques(df,dict):

    print("***************BATCH************")

    for d in dict:
        res = df[d].unique()
        dict[d] |= set(res)


def main():

    chunksize = 1000000

    # name of all the coloumns
    inputColumn = ["Lable", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13", "C1",
                   "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
                   "C15", "C16", "C17", "C18", "C19",
                   "C20", "C21", "C22", "C23", "C24", "C25", "C26"]

    dict = {"C1":set_C1,"C2":set_C2,"C3":set_C3,"C4":set_C4,"C5":set_C5,"C6":set_C6,"C7":set_C7,"C8":set_C8,"C9":set_C9,
           "C10":set_C10,"C11":set_C11,"C12":set_C12,"C13":set_C13,"C14":set_C14,"C15":set_C15,"C16":set_C16,
            "C17":set_C17,"C18":set_C18,"C19":set_C19,"C20":set_C20,"C21":set_C21
          ,"C22":set_C22,"C23":set_C23,"C24":set_C24,"C25":set_C25,"C26":set_C26}

    for chunk in pd.read_csv("/home/ysbhopal11/outputfile.csv", iterator=True, chunksize=chunksize, header=None,names=inputColumn):
        finduniques(chunk, dict)

    f = open('unique.txt', 'w')


    for d in dict:

        string = "Unique features in coloumn "+d+" : "+str(len(dict[d]))
        f.write(string+"\n")
    f.close()


if __name__ == '__main__':
    main()
