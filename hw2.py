import numpy as np
import pandas as pd
import sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

input_file = "supermarket_sales.csv"


data = pd.read_csv(input_file, header = 0)
data.Branch = pd.factorize(data.Branch)[0]+1
data.City = pd.factorize(data.City)[0]+1
data.Customer_type = pd.factorize(data.Customer_type)[0]+1
data.Gender = pd.factorize(data.Gender)[0]+1
data.Product_line = pd.factorize(data.Product_line)[0]+1
data.Payment = pd.factorize(data.Payment)[0]+1


features = list(data.columns[:14])
#print(features)
y = data['Ans']                   # 變出 y 資料
X = data[features]    # 變出 X 資料，將 type 丟棄
#print(X)
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
y_pred = classifier.fit(X_train, y_train).predict(X_test)
print("DecisionTree -> Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
tree.plot_tree(classifier)
#print(type(y[0]))


gnb = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Naive_bayes -> Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))



clf = RandomForestClassifier(max_depth=2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
#clf.fit(X, y)
y_pred = clf.fit(X_train, y_train).predict(X_test)
print("RandomForest -> Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))