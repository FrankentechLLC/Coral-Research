import numpy as np # linear algebra
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE


def k_distortions(reduced_text, min_k, max_k, random_state):
    """
    Return distortion of kmeans cluster over range of k values.
    
    To separate the literature, k-means will be run on the vectorized text. 
    Given the number of clusters, k, k-means will categorize each vector by 
    taking the mean distance to a randomly initialized centroid. The 
    centroids are updated iteratively.
    
    Distortion computes the sum of squared distances from each point to its 
    assigned center. When distortion is plotted against k there will be an 
    optimal k value after which decreases in distortion are minimal.
    """
    distortions = []
    for k in range(min_k, max_k):
        k_means = KMeans(
            n_clusters = k, 
            random_state = random_state, 
            n_jobs = -1
        ).fit(X_reduced)
        k_means.fit(X_reduced)
        distortions.append(sum(np.min(
            cdist(
                X_reduced, 
                k_means.cluster_centers_, 
                'euclidean'), 
            axis = 1
        )) / X.shape[0])
    return distortions


def optimal_k(distortions):
    """Return k at elbow of distortion curve."""
    previous_difference = 0
    for k, distortion in distortions:
        difference = distortion - distortions[k+1]
        if difference < previous_difference: return k
        previous_difference = difference
    return distortions[-1]

    
def clusters(reduced_text, k, random_state):
    """Run k-means on a PCA-processed feature vector."""
    kmeans = KMeans(n_clusters = k, random_state = random_state, n_jobs = -1)
    return kmeans.fit_predict(reduced_text)


def embedded_text(X, perplexity, random_state):
    """
    Reduce a high dimensional feature vector of body text to 2 dimensions. 
    
    The body text can be plotted by using these 2 dimensions as x,y 
    coordinates. 

    t-Distributed Stochastic Neighbor Embedding (t-SNE) reduces dimensionality 
    while keeping similar instances close and dissimilar instances apart. 
    It is mostly used for visualization, especially to visualize clusters of 
    instances in high-dimensional space.
    """

    tsne = TSNE(
        verbose = 1, 
        perplexity = perplexity, 
        random_state = random_state
    )
    return tsne.fit_transform(X.toarray())