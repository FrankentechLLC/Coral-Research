from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


def vectorized_text(dataframe, max_features):
    """
    Return pre-processed data in a vector format for clustering. 
    
    For this purpose we will be using tf-idf. This will convert our string 
    formatted data into a measure of how important each word is to the 
    instance out of the literature as a whole.
    
    We will be clustering based off the content of the body text. The maximum 
    number of features will be limited. Only the top 2 ** 12 features will be 
    used, eseentially acting as a noise filter. Additionally, more features 
    cause painfully long runtimes.
    """
    text = dataframe['processed_text'].values
    vectorizer = TfidfVectorizer(max_features = max_features)
    return vectorizer.fit_transform(text)
    
    
def reduced_text(vectorized_text):
    """
    Return data of minimum dimension for 95% variance. 

    Apply Principle Component Analysis (PCA) to vectorized data. Keeping many 
    dimensions with PCA destroys little information but removes noise/outliers 
    to ease clustering. Note that X_reduced will only be used for k-means, 
    t-SNE will still use the original feature vector X that was generated 
    through tf-idf on the NLP processed text.

    (Thank you Dr. Edward Raff for the suggestion)
    """
    pca = PCA(n_components = 0.95, random_state = 42)
    return pca.fit_transform(vectorized_text.toarray())