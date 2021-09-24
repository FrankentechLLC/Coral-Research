# The labeled plot gives better insight into how the papers are grouped. 
# It is interesting that both k-means and t-SNE are able to agree on certain
# clusters even though they were ran independently. The location of each paper 
# on the plot was determined by t-SNE while the label (color) was determined 
# by k-means. If we look at a particular part of the plot where t-SNE has 
# grouped many articles forming a cluster, it is likely that k-means is 
# uniform in the labeling of this cluster (most of the cluster is the same 
# color). This behavior shows that structure within the literature can be 
# observed and measured to some extent. 

# Now there are other cases where the colored labels (k-means) are spread 
# out on the plot (t-SNE). This is a result of t-SNE and k-means finding 
# different connections in the higher dimensional data. The topics of 
# these papers often intersect so it hard to cleanly separate them. This 
# effect can be observed in the formation of subclusters on the plot. These 
# subclusters are a conglomeration of different k-means labels but may share 
# some connection determined by t-SNE.

# This organization of the data does not act as a simple search engine. The 
# clustering + dimensionality reduction is performed on the mathematical 
# similarities of the publications. As an unsupervised approach, the 
# algorithms may even find connections that were unnaparent to humans. This 
# may highlight hidden shared information and advance further research.


# Topic Modeling on Each Cluster
# 
# Now we will attempt to find the most significant words in each clusters. 
# K-means clustered the articles but did not label the topics. Through 
# topic modeling we will find out what the most important terms for each 
# cluster are. This will add more meaning to the cluster by giving 
# keywords to quickly identify the themes of the cluster.

# For topic modeling, we will use LDA (Latent Dirichlet Allocation). In LDA,
# each document can be described by a distribution of topics and each topic 
# can be described by a distribution of words[.](https://towardsdatascience.com
# /light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-
# allocation-437c81220158)

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def selected_topics(model, vectorizer, top_n = 3):
    """Returns keywords for each topic."""
    current_words, keywords = [], []
    for topic in model.components_:
        words = [(vectorizer.get_feature_names(s)[i], topic[i]) 
                 for i in topic.argsort()[:-top_n - 1:-1]]
        for word in words:
            if word[0] not in current_words:
                keywords.append(word)
                current_words.append(word[0])
                
    keywords.sort(key = lambda x: x[1])  
    keywords.reverse()
    return [i[0] for i in keywords]


def cluster_keywords(
    dataframe, 
    k, 
    min_df = 5, 
    max_df = 0.9, 
    random_state = 42,
    topics_per_cluster = 20
):
    # create one vectorizer for each cluster label
    vectorizers = []
    for ii in range(0, k):
        vectorizers.append(CountVectorizer(
            min_df = min_df, 
            max_df = max_df, 
            stop_words = 'english', 
            lowercase = True, 
            token_pattern = '[a-zA-Z\-][a-zA-Z\-]{2,}'
        ))

    # vectorize the data from each cluster
    vectorized_data = []
    for current_cluster, cvec in enumerate(vectorizers):
        try:
            vectorized_data.append(cvec.fit_transform(
                dataframe.loc[df['y'] == current_cluster, 'processed_text']))
        except:
            print("Not enough instances in cluster: " + str(current_cluster))
            vectorized_data.append(None)

    # Topic modeling is be done with Latent Dirichlet Allocation (LDA), 
    # a generative statistical model allowing sets of words to be explained by 
    # a shared topic

    lda_models = []
    for ii in range(0, k):
        # Latent Dirichlet Allocation Model
        lda = LatentDirichletAllocation(
            n_components = topics_per_cluster, 
            max_iter = 10, 
            learning_method='online',
            verbose = False, 
            random_state = random_state
        )
        lda_models.append(lda)

    # For each cluster, we had created a correspoding LDA model in the 
    # previous step. We will now fit_transform all the LDA models on their 
    # respective cluster vectors

    clusters_lda_data = []

    for current_cluster, lda in enumerate(lda_models):
        if vectorized_data[current_cluster] != None:
            clusters_lda_data.append(
                (lda.fit_transform(vectorized_data[current_cluster]))
            )


    # Append list of keywords for a single cluster to 2D list of length
    # NUM_TOPICS_PER_CLUSTER

    keywords = []
    for vectorizer, lda in enumerate(lda_models):
        if vectorized_data[vectorizer] != None:
            keywords.append(selected_topics(lda, vectorizers[vectorizer]))
            
    return keywords