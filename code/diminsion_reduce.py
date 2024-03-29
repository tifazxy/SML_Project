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

# X_train = de_dimontion_pca(X_train, components=20)