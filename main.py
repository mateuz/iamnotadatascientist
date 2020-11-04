import numpy as np   # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
from plotly.offline import iplot #init_notebook_mode
#init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools

def tuning():

  # Params
  # C - contrabalança a classificação incorreta de exemplos
  # de treinamento pela simplicidade da superfície de decisão.
  # Um C baixo torna a superfície de decisão suave, enquanto 
  # um C alto visa classificar todos os exemplos de treinamento
  # corretamente
  # 
  # Gamma - define quanta influência um único exemplo de 
  # treinamento tem. Quanto maior for a gama, mais próximos 
  # os outros exemplos devem estar para serem afetados.
  #
  # 1.4.6.1. Parameters of the RBF Kernel
  # https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
  # Proper choice of C and gamma is critical to the SVM’s performance.
  # One is advised to use sklearn.model_selection.GridSearchCV with 
  # C and gamma spaced exponentially far apart to choose good values.

  df = pd.read_csv("data/mushrooms.csv")

  features = df.select_dtypes(include="object").columns

  encoder = preprocessing.LabelEncoder() 

  for feat in features:
    df[feat] = encoder.fit_transform(df[feat].astype(str))

  X = df.drop(['class', 'stalk-root'], axis = 1)
  Y = df['class']

  (X_train, X_test, y_train, y_test) = train_test_split(X, Y, test_size = 0.20, random_state = 25)

  clf = svm.SVC()
  param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5],'C': [1, 10, 25, 50, 75, 100]}, {'kernel': ['linear'], 'C': [1, 10, 25, 50, 100]}]
  grid = GridSearchCV(clf, param_grid, cv = 10, scoring = 'accuracy')

  print("Tuning hyper-parameters")
  grid.fit(X_train,y_train)
  print(grid.best_params_)
  print(np.round(grid.best_score_,3))

def main():
  # read data file
  df = pd.read_csv("data/mushrooms.csv")
  
  df.head(6)
  df.tail(6)

  df.info()

  # label encoding for single column
  print(df["class"].value_counts())

  # Fetch features
  features = df.select_dtypes(include="object").columns

  # label encoding for entire dataset
  encoder = preprocessing.LabelEncoder() 

  # loop over 23 cols i.e the features set
  for feat in features:
    # assign unique values for features i.e cols
    df[feat] = encoder.fit_transform(df[feat].astype(str))

  print(df)

  # dropping class form level 1 i.e x=features
  X = df.drop(['class', 'stalk-root'], axis = 1) 
  Y = df['class']

  (X_train, X_test, y_train, y_test) = train_test_split(X, Y, test_size=0.20, random_state=25)
  clf = svm.SVC(C = 100, gamma = 0.001, kernel = 'rbf')
  clf.fit(X_train, y_train)
  predicted = clf.predict(X_test)

  print("Conf. Matrix: \n", confusion_matrix(y_test,predicted))  
  print("Accuracy: ", accuracy_score(y_test, predicted))
  print("Recall: ", recall_score(y_test, predicted))
  print("F1 Score:", f1_score(y_test, predicted))
  
  # predict and compare
  score = clf.score(X_test, predicted)
  print("Mean Accuracy: ", score)
  
  # pred = cross_val_predict(clf, X, Y, cv=10, n_jobs=-1)
  # conf_mat = confusion_matrix(Y, pred)
  # print(conf_mat)

if __name__ == "__main__":
  #tuning()
  main()
  # cap_analysis()
  # pop_habitat_analysis()
  # odor_analysis()