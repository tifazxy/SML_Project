# reduce features
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import time
import pickle

with open('../data/processed_data/data_v2.pkl', 'rb') as f:
    data_reload = pickle.load(f)

X_train = data_reload['X_train']
X_test  = data_reload['X_test']
X_val   = data_reload['X_val']
y_train = data_reload['y_train']
y_test  = data_reload['y_test']
y_val   = data_reload['y_val']


def de_dimention_pca(X, components=2):
    print(f"PCA components:{components}")
    # Apply PCA
    start_time = time.time()
    pca = PCA(n_components=components)
    X_pca = pca.fit_transform(X)
    print("Number of features after PCA:", X_pca.shape[1], X_pca)
    print("--- %s seconds ---" % (time.time() - start_time))
    return X_pca, pca

def de_dimention_tsne(X, components=2, pplexity=2):
    print(f"TSNE components:{components}, pplexity:{pplexity}")
    # Apply t-SNE
    start_time = time.time()
    tsne = TSNE(n_components=components, perplexity=pplexity)
    X_tsne = tsne.fit_transform(X)
    print("Number of features after t-SNE:", X_tsne.shape[1], X_tsne)
    print("--- %s seconds ---" % (time.time() - start_time))
    return X_tsne, tsne

# LDA is the generalized FDA, and FDA is specific for the binary classification tasks, which n_component=1
def de_dimention_fda(X, y, components=1):
    print(f"FDA components:{components}")
    start_time = time.time()
    lda = LDA(n_components=components)
    X_lda = lda.fit_transform(X, y)
    print("X_lda.shape[1] FDA:", X_lda.shape[1], X_lda)
    print("--- %s seconds ---" % (time.time() - start_time))
    return X_lda, lda


# scree plot For choosing n components.
def draw_scree_plot(X):
    fig, axe = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
    i = 0
    for item in [50, 100, 200]:
        start_time = time.time()
        pca = PCA(n_components=item)
        pca.fit_transform(X)
        print("--- %s seconds ---" % (time.time() - start_time))
        # Plot explained variance ratio
        axe[i].plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', label=f'n_components={item}')
        axe[i].legend()
        i+=1
    fig.suptitle('PCA Scree Plot')
    fig.supxlabel('Principal Component')
    fig.supylabel('Explained Variance Ratio')
    plt.tight_layout()
    plt.show()

def draw_dimensionality_reduction(X, y):
    X_pca_train, _ = de_dimention_pca(X, components=20)
    X_tsne_train, _ = de_dimention_tsne(X, components=2)
    X_fda_train, _ = de_dimention_fda(X, y, components=1)
    # Plot the results
    plt.figure(figsize=(12, 4))

    plt.subplot(1,3,1)
    plt.scatter(X_pca_train[:,0], X_pca_train[:,1], c=y, cmap='viridis', alpha=0.5)
    plt.title('PCA')
    
    plt.subplot(1,3,2)
    plt.scatter(X_tsne_train[:,0], X_tsne_train[:,1], c=y, cmap='viridis', alpha=0.5)
    plt.title('t-SNE')
    # plt.title('t-SNE Visualization of Spam Email Dataset')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar()
    
    # plt.subplot(1,3,3)
    plt.scatter(X_fda_train, [0] * len(X_fda_train), c=y, cmap='viridis', alpha=0.5)
    plt.yticks([])
    plt.xlabel('FDA Axis')
    plt.title('Fisher\'s Discriminant Analysis (FDA)')

    plt.tight_layout()
    plt.show()
    return

# draw_scree_plot(X_train.toarray())
# based on the plot we choose 20(or may be 40)

# draw_dimensionality_reduction(X_train.toarray(), y_train)

# num_rows, num_cols = X_train.shape
# print("Number of rows:", num_rows)
# print("Number of columns:", num_cols)

pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train.toarray())
X_test_pca = pca.transform(X_test.toarray())
X_val_pca = pca.transform(X_val.toarray())

data = {
    'X_train': X_train_pca,
    'X_test': X_test_pca,
    'X_val': X_val_pca,
    'y_train': y_train,
    'y_test': y_test,
    'y_val': y_val,
    'pca_model': pca
}

with open('../data/processed_data/data_processed_pca.pkl', 'wb') as f:
    pickle.dump(data, f)
