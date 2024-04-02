import pickle
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# data_v2 tokenized data
# data_processed_pca tokenized and also dimension reduced data
with open('../data/processed_data/data_v2.pkl', 'rb') as f:
    data_reload = pickle.load(f)

X_train = data_reload['X_train']
X_test  = data_reload['X_test']
X_val   = data_reload['X_val']
y_train = data_reload['y_train']
y_test  = data_reload['y_test']
y_val   = data_reload['y_val']

with open('../data/processed_data/data_processed_pca.pkl', 'rb') as f:
    data_reload_pca = pickle.load(f)

X_train_pca = data_reload_pca['X_train']
X_test_pca  = data_reload_pca['X_test']
X_val_pca   = data_reload_pca['X_val']
y_train_pca = data_reload_pca['y_train']
y_test_pca  = data_reload_pca['y_test']
y_val_pca   = data_reload_pca['y_val']

with open('../data/processed_data/data_processed_pca_hard_ham.pkl', 'rb') as f:
    data_reload_hard_ham_pca = pickle.load(f)

X_train_hard_ham_pca = data_reload_hard_ham_pca['X_train']
X_test_hard_ham_pca  = data_reload_hard_ham_pca['X_test']
X_val_hard_ham_pca   = data_reload_hard_ham_pca['X_val']
y_train_hard_ham_pca = data_reload_hard_ham_pca['y_train']
y_test_hard_ham_pca  = data_reload_hard_ham_pca['y_test']
y_val_hard_ham_pca   = data_reload_hard_ham_pca['y_val']

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


# Another model
param_dist_svc = {
    'C': randint(1, 100),            # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel type
    'gamma': ['scale', 'auto'],      # Kernel coefficient
    'degree': randint(1, 10),        # Degree of the polynomial kernel
}
# # try predict
# def cv_svc(X_train, y_train):
#     cv_scores = cross_val_score(svc, X_train, y_train, cv=10, scoring='accuracy')
#     print("Cross-Validation Scores:", cv_scores)
#     print("Mean CV Accuracy:", cv_scores.mean())

# def training_svc(X_train, y_train):
#     svc.fit(X_train, y_train)

# def validation_svc(X_val, y_val):
#     y_val_pred = svc.predict(X_val)
#     val_accuracy = accuracy_score(y_val, y_val_pred)
#     print("Validation Accuracy:", val_accuracy)

# def prediction_svc(X_test, y_test):
#     y_pred = svc.predict(X_test)
#     test_accuracy = accuracy_score(y_test, y_pred)
#     print("Test Accuracy:", test_accuracy)

def workflow(model, param_dist, X_train, y_train, X_val, y_val, X_test, y_test):
    start_time = time.time()
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_jobs=10, n_iter=10, cv=5)
    random_search.fit(X_val, y_val)
    print("--- %s seconds ---" % (time.time() - start_time))
    # Get the best hyperparameters
    best_params = random_search.best_params_
    print("Best Hyperparameters:", best_params)
    # Train the model
    best_model = random_search.best_estimator_
    best_model.fit(X_train, y_train)
    # Predict
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", test_accuracy)

print("==================================")
print("SVC: tokenized dataset:")
workflow(SVC(), param_dist_svc, X_train, y_train, X_val, y_val, X_test, y_test)
print("==================================")
print("SVC: PCA processed dataset:")
workflow(SVC(), param_dist_svc, X_train_pca, y_train_pca, X_val_pca, y_val_pca, X_test_pca, y_test_pca)
print("==================================")
print("SVC: PCA processed hard ham dataset:")
workflow(SVC(), param_dist_svc, X_train_hard_ham_pca, y_train_hard_ham_pca, X_val_hard_ham_pca, y_val_hard_ham_pca, X_test_hard_ham_pca, y_test_hard_ham_pca)

# Random Forest
param_dist_rf = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20],
    'min_samples_split': randint(2, 11)
}
# def training_rf(X_train, y_train):
#     rf_classifier.fit(X_train, y_train)

# def validation_rf(X_val, y_val):
#     y_val_pred = rf_classifier.predict(X_val)
#     val_accuracy = accuracy_score(y_val, y_val_pred)
#     print("Validation Accuracy:", val_accuracy)

# def prediction_rf(X_test, y_test):
#     y_pred = rf_classifier.predict(X_test)
#     test_accuracy = accuracy_score(y_test, y_pred)
#     print("Test Accuracy:", test_accuracy)

print("==================================")
print("Random Forest: tokenized dataset:")
workflow(RandomForestClassifier(n_estimators=100, random_state=42), param_dist_rf, X_train, y_train, X_val, y_val, X_test, y_test)
print("==================================")
print("Random Forest: PCA processed dataset:")
workflow(RandomForestClassifier(n_estimators=100, random_state=42), param_dist_rf, X_train_pca, y_train_pca, X_val_pca, y_val_pca, X_test_pca, y_test_pca)
print("==================================")
print("Random Forest: PCA processed hard ham dataset:")
workflow(RandomForestClassifier(n_estimators=100, random_state=42), param_dist_rf, X_train_hard_ham_pca, y_train_hard_ham_pca, X_val_hard_ham_pca, y_val_hard_ham_pca, X_test_hard_ham_pca, y_test_hard_ham_pca)

# Overfitting analysis