import pickle
import pandas as pd
from DataProcessing import read_to_str
from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata

spam_set_dir = '../../data/spam'
LABEL_SPAM = 1
tokenized_spam_emails, spam_labels = read_to_str(spam_set_dir, LABEL_SPAM)
tfidf_vectorizer = TfidfVectorizer(min_df=3, max_df=0.95)
X_train_tfidf = tfidf_vectorizer.fit_transform(tokenized_spam_emails)
tfidf_arrays = X_train_tfidf.toarray()
# Get feature names
feature_names = tfidf_vectorizer.get_feature_names_out()


def is_english(c):
    return c.isalpha() and unicodedata.name(c).startswith(('LATIN', 'COMMON'))

tfidf_df = pd.DataFrame(tfidf_arrays, columns=feature_names)
df = pd.DataFrame(tfidf_df)
# ls = list(df.columns.values)
# print(ls[0:20])

# print(df.head())
print(df[['official', 'oppressive', 'opps', 'patient', 'pc0b2tbt7n7dln', 'query', 'quest', 'question', 'questionable','r03kofmcj','resdir', 'research', 'researched', 'researcher', 's_img', 'sa', 'sacrament', 'sacramento', 'tommy', 'tomorrow']].head())

# with open('../data/processed_data/data_processed_pca.pkl', 'rb') as f:
#     data_reload = pickle.load(f)
# X_train = data_reload['X_train']
# X_test  = data_reload['X_test']
# X_val   = data_reload['X_val']
# y_train = data_reload['y_train']
# y_test  = data_reload['y_test']
# y_val   = data_reload['y_val']
# df = pd.DataFrame(X_train)
# print(list(df.columns.values))
# print(df.head())