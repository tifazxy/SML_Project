import pickle
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

with open('./data/processed_data/data.pkl', 'rb') as f:
    data_reload = pickle.load(f)

X_train = data_reload['X_train']
X_test  = data_reload['X_test']
X_val   = data_reload['X_val']
y_train = data_reload['y_train']
y_test  = data_reload['y_test']
y_val   = data_reload['y_val']

# Another model
svc = SVC()

# try predict
def cv_svc(X_train, y_train):
    cv_scores = cross_val_score(svc, X_train, y_train, cv=10, scoring='accuracy')
    print("Cross-Validation Scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())

def training_svc(X_train, y_train):
    svc.fit(X_train, y_train)

def validation_svc(X_val, y_val):
    y_val_pred = svc.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print("Validation Accuracy:", val_accuracy)

def prediction_svc(X_test, y_test):
    y_pred = svc.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", test_accuracy)

cv_svc(X_train, y_train)
training_svc(X_train, y_train)

validation_svc(X_val, y_val)

prediction_svc(X_test, y_test)

# Define the baseline model(most_frequent class classifier)
baseline_model = DummyClassifier(strategy='most_frequent')


# try predict
def cv_baseline(X_train, y_train):
    cv_scores = cross_val_score(baseline_model, X_train, y_train, cv=10, scoring='accuracy')
    print("Cross-Validation Scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())

def training_baseline(X_train, y_train):
    baseline_model.fit(X_train, y_train)

def validation_baseline(X_val, y_val):
    y_val_pred = baseline_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print("Validation Accuracy:", val_accuracy)

def prediction_baseline(X_test, y_test):
    y_pred = baseline_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", test_accuracy)

cv_baseline(X_train, y_train)
training_baseline(X_train, y_train)

validation_baseline(X_val, y_val)

prediction_baseline(X_test, y_test)

import numpy as np
print("Training Label distribution:", np.bincount(y_train)/ len(y_train))
print("Validation Label distribution:", np.bincount(y_val)/ len(y_val))
print("Testing Label distribution:", np.bincount(y_test)/ len(y_test))