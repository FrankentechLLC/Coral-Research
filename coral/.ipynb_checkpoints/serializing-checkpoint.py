import pickle


def serialize_intermediates_outputs(
    cluster_keywords, 
    dataframe, 
    embedded_text, 
    predicted_clustering
):
    """
    Save current outputs to file

    Re-running some parts of the notebook (especially vectorization and t-SNE) 
    are time intensive tasks. We want to make sure that the important outputs 
    for generating the bokeh plot are saved for future use.
    """
    with open('lib/topics.txt','w') as f:
        count = 0
        for cluster_keyword in cluster_keywords:
            if vectorized_data[count] != None: 
                f.write(', '.join(cluster_keyword) + "\n")
            else:
                f.write("Not enough instances to be determined. \n")
                f.write(', '.join(cluster_keyword) + "\n")
            count += 1
            
    # save the COVID-19 DataFrame, too large for github
    pickle.dump(dataframe, open("plot_data/dataframe.p", "wb"))

    # save the final t-SNE
    pickle.dump(
        embedded_text, open("plot_data/embedded_text.p", "wb")
    )

    # save the labels generate with k-means(20)
    pickle.dump(
        predicted_clustering, open("plot_data/predicted_clustering.p", "wb")
    )