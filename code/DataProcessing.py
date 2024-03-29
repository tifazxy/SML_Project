import os
import chardet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# Run once
# nltk.download('punkt') run this once in code or envirment to down load the tokenlizer which is essential for word_tokenize
# nltk.download('stopwords')

spam_set_dir = '../../data/spam'
ham_set_dir = '../../data/ham'
LABEL_SPAM = 1
LABEL_HAM = 0

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']
    return encoding

def read_to_str(dir, label):
    # Initialize lists to store tokenized emails and labels
    tokenized_emails = []
    labels = []
    for dir_name in os.listdir(dir):
        print(os.path.join(dir, dir_name))
        for filename in os.listdir(os.path.join(dir, dir_name)):
            file_path = os.path.join(os.path.join(dir, dir_name), filename)
            encoding = detect_encoding(file_path)
            try:
                # use latin-1 to decode is make sence, cannot use utf-8 because it's not adaptation the encoding.
                with open(file_path, 'r', encoding=encoding) as file:
                    email_content = file.read()
                    # Tokenization & Lowercasing: make sure same word be treat as same one
                    tokens = word_tokenize(email_content.lower())
                    # Removing Stopwords:
                    stopwords_list = set(stopwords.words('english'))
                    tokens = [word for word in tokens if word not in stopwords_list]
                    # Finish processing, add to list
                    tokenized_emails.append(' '.join(tokens))
                    # Also store the label
                    labels.append(label)
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    return tokenized_emails, labels

tokenized_spam_emails, spam_labels = read_to_str(spam_set_dir, LABEL_SPAM)
tokenized_ham_emails, ham_labels = read_to_str(ham_set_dir, LABEL_HAM)

# Combine spam and ham data
tokenized_emails = tokenized_spam_emails + tokenized_ham_emails
labels = spam_labels + ham_labels

# From here split training, validation, testing sets. The main consideration is we don't have access of testing data before the completement of the model.
# Same dataprocessing should do on the test sets later aftering training.
# 70% training, 15% validation, 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(tokenized_emails, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Testing set: {len(X_test)} samples")

# exclude words that appear in only 4 document
# exclude words that appear in more than 95% of the documents. [8853 rows x 98265 columns]
tfidf_vectorizer = TfidfVectorizer(min_df=3, max_df=0.95)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

import pickle
data = {
    'X_train': X_train_tfidf,
    'X_test': X_test_tfidf,
    'X_val': X_val_tfidf,
    'y_train': y_train,
    'y_test': y_test,
    'y_val': y_val
}

with open('../data/processed_data/data_v2.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Data Label distribution:", len(spam_labels)/ len(labels))
print("Data Label num:", len(spam_labels))