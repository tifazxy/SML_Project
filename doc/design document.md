# Design document

## Define the Objective:

Compare the effectiveness of PCA, t-SNE and FDA in reducing the dimensionality of a dataset while preserving importand information.

## Select the Dataset:
The dataset should have a high-dimensional feature space to demonstrate the effectiveness of dimensionality reduction teniques. Thus we use spam emails.

## Preprocess the Dataset:
Removing any irrelevant features, handling missing values, scaling the data if necessary. 

We apply the tokenize, lower case converter, stop words removing process on the original dataset.

Apply Dimensinality Reductioin Techniques:
We applied PCA, t-SNE, and FDA to othe preprocessed dataset. For each technique, we experimented with different parameters, e.g.:
- number of components for PCA
- perplexity for t-SNE
to see how they affect the results.

Visualize the Results:
We visualized the reduced-dimensional data using scatter plots

## Novelty & Scholarly significance
Try out best to find the novelty, come up with it, otherwise will lost 1-2 points

## Open problems in existing literature
How to find? Maybe some papers has also compare the algorhms, we can find what the difference between from our work and theirs.
