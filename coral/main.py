"""
Paper Cluster 

Helps researchers sift through a literature collection by representing it with 
an interactive scatterplot that clusters publications about highly similar 
topics under shared labels.  Users can see the whole plot or filter by cluster 
or keyword.  Hovering over plot points displays basic information like title, 
author, journal, and abstract, and clicking on a point presents a menu with 
the publication URL.
"""


import loading
import processing
import feature_engineering
import clustering
import labelling
import serializing
import evaluating
import plotting


def main():
    # configure
    with open("config.txt") as f:
        entries = [line[:-1].split(" = ") for line in f.readlines()]
    config = {entry[0] : entry[1] for entry in entries}
    for key in config:
        if "path" in key: pass
        elif "." in config[key]: config[key] = float(config[key])
        else: config[key] = int(config[key])
        
    
    # load
    metadata_path = f'{root_path}/{config["metadata filename"]}'
    meta_dataframe = loading.meta_dataframe(metadata_path)
    json_paths = loading.json_paths(config["root_path"])
    dataframe = loading.paper_dataframe(json_paths)
    
    # process
    processing.engineer_features(dataframe)
    processing.select_english_articles(dataframe)
    processing.process_text(dataframe, config["most words considered"])
    vectorized_text = feature_engineering.vectorized_text(
        dataframe, 
        config["most features"]
    )
    reduced_text = feature_engineering.reduced_text(dataframe)
    
    # cluster
    distortions = clustering.k_distortions(
        reduced_text,
        config["fewest clusters"],
        config["most clusters"],
        config["random seed"]
    )
    k = clustering.optimal_k(distortions)
    clusters = clustering.clusters(reduced_text, k, random_state)
    embedded_text = clustering.embedded_text(
        vectorized_text, 
        config["perplexity"],
        config["random seed"],
    )
    
    # label
    cluster_keywords = labelling.cluster_keywords(dataframe, k)
    
    # serialize
    serialize_intermediates_outputs(
        cluster_keywords, 
        dataframe, 
        embedded_text, 
        clusters
    )
    
    # evaluate
    evaluating.test_clustering(
        vectorized_text, 
        clusters, 
        config["random seed"],
        config["test size"],
        config["iterations"]
    )
    
    # plot
    cluster_plot = plotting.cluster_plot(embedded_text, clusters, k)
    plotting.interactive_plot(
        embedded_text, 
        clusters, 
        config["random seed"],
        config["plot width"],
        config["plot height"]
    )
    
if __name__ == "__main__": main()