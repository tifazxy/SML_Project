# Code Analysis

## PCA, t-SNE and FDA

Q1: Can I use fit_transform for all my training, validation, and testing data? Or should I only use transform on my validation and testing data?
- It’s common practice to use ‘fit_transform’ on the training data to learn the transformation parameters(e.g., principal components) and then use ‘transform’ on the validation and testing data to apply the learned transformation.
- Using ‘fit_transform’ on the validation and testing data could lead to data leakage , as the transformation parameters would be influenced by the entire dataset, including the validation and testing parts.

Q2: Should I use the same instance of PCA to apply transform on my test data or I can create a new instance with same parameter to apply transform on my test data? What’s the difference?
- use same instance

Q3: So for the plot, because the data is tokenized spam email data, does the X_pca[:,0] means one of the classes? If it means the first principal component, because I use n_component=20, so in the figure select the top 2 principal components X_pca[:,0], X_pca[:,1]? 
- Yes the plot shows the data projected onto the top 2 principal components, which capture the most variance among the 20 components.
- the principal component, which is a linear combination of the original features that captures the most variance in the data.

Q4: Is it possible that I present 20 components in the figure?
- One approach could be to create a series of subplots, each showing the data projected onto a different pair of principal components(e.g., PC1 vs. PC2, PC1 vs. PC3, etc.)
-Another approach could be to use techniques like heatmap or clustering to visualize the relative importance of each component in capturing the variance in the data.

Q5: After the transform, what's the data represent?
- After the transformation, the data is indeed 2D, with each row still representing an email instance.
- However, the columns represent the new “features” in the transformed space, which are the principal components, not the original features of the dataset.
- The values in each column represent the projection of the original data onto that principal component, capturing the variance of the data along that direction rather than the original feature values. 

Q6: Explain the PCA plot.
- The two different colors representing the specific classes show how the data is separated or grouped in the reduced 2D space.
- For principal component 0: both classes are spread out along the first principal component in a similar manner.
- For principal component 1: these classes are spread along the second principal component, with one class tending towards positive values and the other towards negative values.


Q7: What does the value in the plot mean? Variance? what does negative variance in the plot mean?
- The values do not represent variance directly. Instead, they represent the projection of the data points onto the corresponding principal components. Since PCA finds orthogonal axes in the data that capture the most variance, the direction of these axes is arbitrary and can be positive or negative. 
- The sign of the values along the axes simply indicates the orientation of the component in the original feature space.

Q8: pca.explained_variance_ratio_
- It calculates the ratio of variance explained by each of the principal components.
When we’re calling pca.explained_variance_ratio_, the values in the resulting array sum to 1, as they represent the proportion of the total variance explained by each principal component.

Q9: Why t-SNE is not used as a dimensionality reduction method?
- t-SNE is a  non-linear algorithm. which means it doesn’t preserve distances or relationships between data points in the original high-dimensional space.

Q10: Does pca and FDA linear?
- PCA (Principal Component Analysis) is a linear technique, meaning that it seeks to find linear combinations of the original features that capture the most variance in the data. The principal components are orthogonal to each other and are linear transformations of the original features.
- On the other hand, FDA (Fisher's Discriminant Analysis) is also linear, but it differs from PCA in that it considers class labels in addition to the variance of the data. FDA aims to find a linear combination of features that best separates the classes in the data. The resulting components (Fisher discriminants) are also linear transformations of the original features.
- Both PCA and FDA are linear techniques, but they serve different purposes: PCA is primarily used for dimensionality reduction and capturing variance, while FDA is used for supervised dimensionality reduction and class separation.
