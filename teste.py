import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from geneticalgorithm import geneticalgorithm as ga

df = pd.read_csv("data/mushrooms.csv")

features = df.select_dtypes(include="object").columns

encoder = preprocessing.LabelEncoder()
for feat in features:
    df[feat] = encoder.fit_transform(df[feat].astype(str))

data = df.drop(['class','stalk-root'], axis = 1)
labels = df['class']

# solutions = 100
# dimension = 21
# shape = (solutions, dimension)

# population = np.random.randint(low = 0, high = 2, size = shape)

# print(population.shape)
# print(population)

feats = features.drop(['class','stalk-root'])
# calcPopulationFitness(population, feats, X, Y)

def reduceFeatures(solution,features):
    indexs = np.where(solution == 1)[0]
    reducedFeatures = [features[i] for i in indexs] 
    return reducedFeatures

def calcPopulationFitness(population, features, data, labels):
    for current in population:
        selectedFeatures = reduceFeatures(current, features)
        filteredData = data.filter(items = selectedFeatures)

        # print( current )
        # print( selectedFeatures )
        # print( filteredData )

        (xTrain, xTest, yTrain, yTest) = train_test_split(filteredData, labels, test_size = 0.20, random_state = 25)
        clf = svm.SVC(C = 100, gamma = 0.001, kernel = 'rbf')
        clf.fit(xTrain, yTrain)
        predicted = clf.predict(xTest)
        print("F1 Score:", f1_score(yTest, predicted))

def fitness(X):
    selectedFeatures = reduceFeatures(X, feats)
    filteredData = data.filter(items = selectedFeatures)

    if filteredData.empty:
        return 0
    
    (xTrain, xTest, yTrain, yTest) = train_test_split(filteredData, labels, test_size = 0.20, random_state = 25)
    clf = svm.SVC(C = 100, gamma = 0.001, kernel = 'rbf')
    clf.fit(xTrain, yTrain)
    predicted = clf.predict(xTest)

    v1 = 21 - np.count_nonzero(X) 
    v2 = accuracy_score(yTest, predicted) * 1000

    return -(v1 + v2)

def runGA():
    params = {'max_num_iteration': 500,\
                   'population_size': 50,\
                   'mutation_probability': 0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type': 'uniform',\
                   'max_iteration_without_improv': None}

    model=ga(function = fitness,\
                dimension = 21,\
                variable_type = 'bool',\
                variable_boundaries = None,\
                algorithm_parameters = params)
    
    print(model.param)
    model.run()

def main():
    df = pd.read_csv("data/mushrooms.csv")

    features = df.select_dtypes(include="object").columns

    encoder = preprocessing.LabelEncoder()
    for feat in features:
        df[feat] = encoder.fit_transform(df[feat].astype(str))

    X = df.drop(['class','stalk-root'], axis = 1)
    Y = df['class']
 
    solutions = 100
    dimension = 21
    shape = (solutions, dimension)
    population = np.random.randint(low = 0, high = 2, size = shape)

    # print(population.shape)
    # print(population)

    feats = features.drop(['class','stalk-root'])
    calcPopulationFitness(population, feats, X, Y)

# main()
runGA()