# Workflow

## Data source:

SpamAssassin https://spamassassin.apache.org/old/publiccorpus/

The difference between ham/spam and ham_2/spam_2 is the “_2” folders are the latest added, when we are processing, we regard ham and ham_2 as ham, the same processing as spam; 

All files are raw emails, we use this dataset as the corpus for the project. In the project, the number of instances is shown below:

easy_ham: 5052 | easy_ham_2: 1401 | spam: 1002 | spam_2: 1398 | 
ham total: 6453 | spam total: 2400 | total: 8853. 
spam ratio 27.11%

## Data processing:(Already finished by 0315)

1. Read raw corpus into memory, apply basic processing techniques, we are using Tokenization, Lowercasing, and Removing Stopwords.  At the same time recording the label for each raw data.
2. Splitting tokenized dataset, 70% training, 15% validation, 15% testing.
Training set: 6197 samples | Validation set: 1328 samples | Testing set: 1328 samples.
3. Construct feature extraction function, in our project, we use TF-IDF score as features, each sample is a vector of its token’s TF-IDF score.
4. Construct Dimensionality reduction functions, to select features. Now we are applying PCA, going to add t-SNE and FDA(LDA).
5. Construct scree plot function, to decide the number of the component of PCA.
6. Build baseline model, using the sklearn DummyClassifier(most_frequent class classifier), since the most frequent class in our data has a 73% occurrence rate (binary around 0.7289:0.2711 prior distribution but based on the different split set), The DummyClassifier will always predict the most frequent class, which corresponds to the majority class in our dataset, resulting in an accuracy score close to 73%. as our baseline accuracy.
7. Apply (5) to training data, found the number of components could be 20; Apply (3) and (4) on training data, get the processed training data.
8. Apply baseline model (6) with training data doing cross validation. Fit training data to the baseline model.
9. Apply (5) to validation data, find the number of components could be 20; Apply (3) and (4) on validation data, get the processed validation data.
10. Predict using baseline model with validation data to see the performance.
11. Apply (5) to test data, find the number of components could be 20; Apply (3) and (4) on test data, get the processed test data.
12. Predict using baseline model with test data to see the performance.

## Next step: (each member complete 2 models with the workflow)

- We need to use more models to complete cross validation(CV), training, validation and predicting on test data. Basically the 8-12 steps, I already preprocessed the steps 7,9,11 and compressed the dataset in to pickle file(based on PCA currently, may add different variance Dimensionality reduction later), you can also load this pickle as train/validation/testing dataset directly.

- Models we’re going to choose Naive Bayes, SVM, AdaBoost, K-Nearest Neighbors, Bagging, Random Forests, logistic regression.

- Analyzing:
    - Virtualize the cross-validation process, to analysis the work routine of CV. (overfitting analysis and do the tune of hyperparameters.)
    - Analysis the performance of the model with metrics as accuracy, precision, recall, F1-score, and ROC. (Maybe we don’t need to use all the metrics. And also, maybe we need virtualize ROC)
    - Compare and analysis the performance with baseline, maybe also analysis the influence of imbalance of dataset.



