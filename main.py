import numpy as np   # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing

def main():

  # read data file
  df = pd.read_csv("data\mushrooms.csv")
  print(df)

  # read first 6 rows
  df.head(6)

  # read last 6 rows
  df.tail(6)

  # read structure of dataframe
  # df in which 8124 rows,23 columns
  # all not null values
  # all datatype-object,object dtype is for all text data; other dtypes-int,float,date-time
  df.info()

  # t1-no null values
  # t2-all data should be in numeric format only

  # label encoding for single column
  df["class"].value_counts() # eg-in column class all "e" and "p" are replaced with numeric format & total=8124

  print(df["class"].value_counts())

  #Fetch features of type object
  objfeatures = df.select_dtypes(include="object").columns
  print(objfeatures)

  #label encoding for entire dataset;
  # iterate a loop for features of type object
  le = preprocessing.LabelEncoder() # it will assign unique values for individual features i.e cols

  # for loop will run for 23 cols i.e range objfeature
  for feat in objfeatures:
    # all "object(str) dtype" will replace by "int dtype"
    df[feat] = le.fit_transform(df[feat].astype(str))
  
  df.info()

  


  # separate x and y i.e features and labels
  # x & y
  X = df.drop(["class"],axis=1) # dropping class form level 1 i.e x=features
  y = df['class'] # explicitly for class y=labels

  X.info()

  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

  X_train.info() # 70%values
  X_test.info()  # 30%values
  

  #model training
  from sklearn.naive_bayes import GaussianNB #gaussian naive bayes

  gnb = GaussianNB() # creating variable of algo and follow rules defined by algo

  gnb.fit(X_train, y_train) # training empty brain i.e for given x these was y i.e understanding 70% data

  

  # predicting Y value
  y_prediction = gnb.predict(X_test)

  print(y_prediction)

  print(y_test)

if __name__ == "__main__":
  main()