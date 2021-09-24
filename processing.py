from tqdm import tqdm
from langdetect import detect
from langdetect import DetectorFactory

### CAUTION: HIDEOUS ABOMINATION
from IPython.utils import io
with io.capture_output() as captured:
    get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz')
### THIS THING NEEDS FIXED
# - Does not appear to check for dependency
# - Version of en_core_sci_lg is static
# - Probably needs code to manage all this
# - Ask Maksim about it
# - Requires iPython -- does it need to?

    
import string
import spacy
import en_core_sci_lg


def engineer_features(dataframe):
    """Adds word count columns for both abstract and body text."""
    dataframe['abstract_word_count'] = dataframe['abstract'].apply(
        lambda x: len(x.strip().split()) # abstract word count 
    )  
    dataframe['body_word_count'] = dataframe['body_text'].apply(
        lambda x: len(x.strip().split()) # body word count
    )
    dataframe['body_unique_words'] = dataframe['body_text'].apply(
        lambda x:len(set(str(x).split())) # body unique word count 
    )

    # The dataframe may include duplicates because authors submitted 
    # their articles to multiple journals.
    # (Thank you Desmond Yeoh for recommending the below approach on Kaggle)
    dataframe.drop_duplicates(['abstract', 'body_text'], inplace=True)

    # Now that we have our dataset loaded, we need to clean-up the text to
    # improve any clustering or classification efforts. First, let's drop Null 
    # vales:
    dataframe.dropna(inplace=True)


def select_english_articles(dataframe):
    """
    Select the English-language articles from the dataframe.
    
    Determines the language of each paper in the dataframe. Not all sources are 
    English, and the language needs to be identified so that we know how handle 
    these instances.
    """

    # set seed
    DetectorFactory.seed = 0

    # hold label - language
    languages = []

    # go through each text
    for ii in tqdm(range(0,len(dataframe))):
        # split by space into list, take the first x intex, join with space
        text = df.iloc[ii]['body_text'].split(" ")

        lang = "en"
        try:
            if len(text) > 50:
                lang = detect(" ".join(text[:50]))
            elif len(text) > 0:
                lang = detect(" ".join(text[:len(text)]))
        # ught... beginning of the document was not in a good format
        except Exception as e:
            all_words = set(text)
            try:
                lang = detect(" ".join(all_words))
            # what!! :( let's see if we can find any text in abstract...
            except Exception as e:

                try: # label it through the abstract then
                    lang = detect(dataframe.iloc[ii]['abstract_summary'])
                except Exception as e:
                    lang = "unknown"
                    pass

        # get the language    
        languages.append(lang)

    languages_dict = {}
    for lang in set(languages): languages_dict[lang] = languages.count(lang)

    dataframe['language'] = languages
    dataframe = dataframe[dataframe['language'] == 'en']


def stopwords():
    """
    Finds and removes stopwords common words that would clutter clustering.
    
    Research papers will often frequently use words that don't actually 
    contribute to the meaning and are not considered everyday stopwords.

    Thank you Daniel Wolffram for the idea.
    Cite: [Custom Stop Words | Topic Modeling: Finding Related Articles]
    https://www.kaggle.com/danielwolffram/topic-modeling-finding-related-articles
    """
    stopwords = list(spacy.lang.en.stop_words.STOP_WORDS)
    
    with open(custom_stop_words_path) as f:
        custom_stop_words = f.readlines()
    
    for w in custom_stop_words: if w not in stopwords: stopwords.append(w)
            
    return stopwords


def spacy_tokenizer(sentence):
    """
    Processes the text data. 

    For this purpose we will be using the spacy library. This function will 
    convert text to lower case, remove punctuation, and find and remove 
    stopwords. For the parser, we will use en_core_sci_lg. This is a model 
    for processing biomedical, scientific or clinical text.
    """
    mytokens = parser(sentence)
    mytokens = [
        word.lemma_.lower().strip() 
        if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens
    ]
    mytokens = [
        word for word in mytokens 
        if word not in stopwords() and word not in string.punctuation
    ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens


def process_text(dataframe, max_length):
    """
    Append a column of processed body text.
    """
    parser = en_core_sci_lg.load(disable=["tagger", "ner"])
    parser.max_length = max_length

    tqdm.pandas()
    dataframe["processed_text"] = dataframe["body_text"].progress_apply(
        spacy_tokenizer, args = [stopwords(), string.punctuation]
    )