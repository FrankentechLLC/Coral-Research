import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json


def meta_dataframe(path):
    """
    Load the metadata of the dateset.
    
    'title' and 'journal' attributes may be useful later when we cluster 
    the articles to see what kinds of articles cluster together.
    """
    return pd.read_csv(path, dtype = {
        'pubmed_id': str,
        'Microsoft Academic Paper ID': str, 
        'doi': str
    })


class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
        self.paper_id = content['paper_id']
        self.abstract = [entry['text'] for entry in content['abstract']]
        self.body_text = [entry['text'] for entry in content['body_text']]
        self.abstract = '\n'.join(self.abstract)
        self.body_text = '\n'.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'


def get_breaks(content, length):
    """
    Add an html break every n words. 
    
    Ensures the hover tool of the interactive plot fits the screen.
    """
    data = ""
    words = content.split(' ')
    total_chars = 0

    # add break every length characters
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            data = data + "<br>" + words[i]
            total_chars = 0
        else:
            data = data + " " + words[i]
    return data


def json_paths(root_path):
    """
    Fetch path to all JSON files.
    """
    return glob.glob(f'{root_path}/**/*.json', recursive = True)


def paper_dataframe(
    json_paths,      
    abstract_snippet_wordcount = 100,         
    line_length = 40
):
    """
    Reads json-formatted articles into a Pandas DataFrame.
    """
    
    data = {
        'paper_id': [], 
        'doi' : [], 
        'abstract' : [], 
        'body_text' : [], 
        'authors' : [], 
        'title' : [], 
        'journal' : [], 
        'abstract_summary' : []
    }
    
    for index, entry in enumerate(json_paths):
        if index % (len(json_paths) // 10) == 0:
            print(f'Processing index: {index} of {len(json_paths)}')

        try: content = FileReader(entry)
        except: continue  # invalid paper format, skip

        meta_dataframe = meta_dataframe(json_paths)
        meta_data = meta_dataframe.loc[meta_df['sha'] == content.paper_id]
        if len(meta_data) == 0: continue # no metadata, skip this paper

        data['abstract'].append(content.abstract)
        data['paper_id'].append(content.paper_id)
        data['body_text'].append(content.body_text)

        # create a column for an abstract snippet to be used in a plot
        if len(content.abstract) == 0: 
            data['abstract_summary'].append("Not provided.")
        elif len(content.abstract.split(' ')) > abstract_snippet_wordcount:
            info = content.abstract.split(' ')[:abstract_snippet_wordcount]
            summary = get_breaks(' '.join(info), line_length)
            data['abstract_summary'].append(summary + "...")
        else:
            summary = get_breaks(content.abstract, line_length)
            data['abstract_summary'].append(summary)

        try:
            authors = meta_data['authors'].values[0].split(';')
            if len(authors) > 2:
                columns['authors'].append(
                    get_breaks('. '.join(authors), line_length)
                )
            else: data['authors'].append(". ".join(authors))
        except: # if only one author - or Null value
            data['authors'].append(meta_data['authors'].values[0])

        try: # add the title information, add breaks when needed
            title = get_breaks(meta_data['title'].values[0], line_length)
            data['title'].append(title)
        except: # if title was not provided
            data['title'].append(meta_data['title'].values[0])

        data['journal'].append(meta_data['journal'].values[0])
        data['doi'].append(meta_data['doi'].values[0])
    
        return pd.DataFrame(data, columns = data.keys)