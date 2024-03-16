import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split


# Run once
# nltk.download('punkt') run this once in code or envirment to down load the tokenlizer which is essential for word_tokenize
# nltk.download('stopwords')

spam_set_dir = 'data/spam'
ham_set_dir = 'data/ham'
LABEL_SPAM = 1
LABEL_HAM = 0

# Initialize lists to store tokenized emails and labels
tokenized_emails = []
labels = [] # ham = 0, spam = 1

def read_to_str(dir, label):
    for dir_name in os.listdir(dir):
        print(os.path.join(dir, dir_name))
        for filename in os.listdir(os.path.join(dir, dir_name)):
            # use latin-1 to decode is make sence, cannot use utf-8 because it's not adaptation the encoding.
            with open(os.path.join(os.path.join(dir, dir_name), filename), 'r', encoding='latin-1') as file:
                email_content = file.read()
                # Tokenization:
                tokens = word_tokenize(email_content)

                # Lowercasing: make sure same word be treat as same one
                tokens = [word.lower() for word in tokens]

                # Removing Stopwords:
                stopwords_list = set(stopwords.words('english'))
                tokens = [word for word in tokens if word not in stopwords_list]

                # Finish processing, add to list
                tokenized_emails.append(' '.join(tokens))
                # Also store the label
                labels.append(label)
                # print(tokens) 


read_to_str(spam_set_dir, LABEL_SPAM)
read_to_str(ham_set_dir, LABEL_HAM)


# print(tokenized_emails)

# From here split training, validation, testing sets. The main consideration is we don't have access of testing data before the completement of the model.
# Same dataprocessing should do on the test sets later aftering training.
# 70% training, 15% validation, 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(tokenized_emails, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Testing set: {len(X_test)} samples")

def create_tfidf_df(X):
    # Initialize TfidfVectorizer, 
    # exclude words that appear in only 4 document
    # exclude words that appear in more than 95% of the documents. [8853 rows x 98265 columns]
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.95)
    # Generate TF-TDF matrix for each email
    tfidf_matrices = tfidf_vectorizer.fit_transform(X)
    # Convert TF-IDF matrices to array
    tfidf_arrays = tfidf_matrices.toarray()
    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()

    tfidf_df = pd.DataFrame(tfidf_arrays, columns=feature_names)
    # tfidf_df['label'] = labels

    # print(tfidf_df)
    # print(tfidf_df.columns.values)
    return tfidf_df

# reduce features
import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def de_dimontion_pca(X, components=2):
    # Apply PCA
    pca = PCA(n_components=components)
    X_pca = pca.fit_transform(X)

    # # Plot the results
    # plt.figure(figsize=(12, 4))

    # plt.subplot(1,3,1)
    # plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis', alpha=0.5)
    # plt.title('PCA')

    # plt.tight_layout()
    # # plt.show()
    # plt.savefig('reduce {}.png'.format('pca')) 

    print("Number of features after PCA:", X_pca.shape[1], X_pca)

    # # See which feature contribute more in PCA
    # # Inspect the leadings of the original features on the first two principal components
    # print("Loadings of original features on the first two principal components:")
    # print(pca.components_[:2, :]) # Print the loadings of the first two principal components
    return X_pca

def de_dimontion_tsne(X, components=2, pplexity=2):
    # Apply t-SNE
    tsne = TSNE(n_components=components, perplexity=pplexity)
    X_tsne = tsne.fit_transform(X)
    
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1,3,3)
    # plt.scatter(X_tsne[:, 0], X_tsne[:,1], c=labels, cmap='viridis', alpha=0.5)
    # plt.title('t-SNE')

    # plt.tight_layout()
    # # plt.show()
    # plt.savefig('reduce {}.png'.format('tsne')) 
    print("Number of features after t-SNE:", X_tsne.shape[1], X_tsne)
    return X_tsne

def de_dimontion_fda(X, y):
    return


# scree plot
import time
def draw_scree_plot(X):
    for item in [100, 500, 1000]:
        pca = PCA(n_components=item)
        pca.fit_transform(X)

        start_time = time.time()
        # Plot explained variance ratio
        plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot')
        # plt.show()
        plt.savefig('Scree_Plot_{}.png'.format(item)) 
        print("--- %s seconds ---" % (time.time() - start_time))


# draw_scree_plot(X_train)
# based on the plot we choose 20(or may be 40)

# X_train = create_tfidf_df(X_train)
# X_train = de_dimontion_pca(X_train, components=20)


# Overfitting analysis
#============================================

# import pickle
# X_val = create_tfidf_df(X_val)
# X_val = de_dimontion_pca(X_val, components=20)
# X_test = create_tfidf_df(X_test)
# X_test = de_dimontion_pca(X_test, components=20)

# data = {
#     'X_train': X_train,
#     'X_test': X_test,
#     'X_val': X_val,
#     'y_train': y_train,
#     'y_test': y_test,
#     'y_val': y_val
# }

# with open('./data/processed_data/data.pkl', 'wb') as f:
#     pickle.dump(data, f)

print("Data Label distribution:", np.bincount(labels)/ len(labels))
print("Data Label num:", np.bincount(labels))