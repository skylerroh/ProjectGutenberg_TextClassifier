from os import listdir
from os.path import isfile, join
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import PorterStemmer
import path
import pandas as pd
import numpy as np
import re
import pdb

"""
Set these variables to be the top level directory of training samples (i.e. your_own_path/Training/) as well as the directory to store the result CSVs
"""
TRAINING_DIR = "/Users/fenglin/Desktop/stat154/Training/"
OUTPUT_DIR = "/Users/fenglin/Desktop/stat154/Training_Output/"

FILTERED_FEATURE_SET_SIZE_TFIDF = 1500
FILTERED_FEATURE_SET_SIZE_WC = 1500
WC_START_AT_INDEX = 100

MAX_FEATURE_LENGTH = 20

GENRE_FILENAMES = ["Child(0)", "History(1)", "Religion(2)", "Science(3)"]

STOP_WORDS = ["a","able","about","across","after","all","almost","also","am","among","an","and","any","are","as","at","be","because","been","but","by","can","cannot","could","dear","did","do","does","either","else","ever","every","for","from","get","got","had","has","have","he","her","hers","him","his","how","however","i","if","in","into","is","it","its","just","least","let","like","likely","may","me","might","most","must","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","she","should","since","so","some","than","that","the","their","them","then","there","these","they","this","tis","to","too","twas","us","wants","was","we","were","what","when","where","which","while","who","whom","why","will","with","would","yet","you","your",
              "project", "gutenberg", "ebook", "title", "author", "release", "chapter"]

stemmer = PorterStemmer()

def tfidf(word, dfs, transformer, doc_word_sums, zero_filled_series):
  """
  Given a word, compute the sum of tf-idf values across all observations in the four genres
  """
  df_word_sum_info = [(df.loc[:,word], doc_word_sum) if (word in df.columns) else (zeros, doc_word_sum) for df, doc_word_sum, zeros in zip(dfs, doc_word_sums, zero_filled_series)]

  freqs = pd.concat([elem[0] for elem in df_word_sum_info])

  word_count_sum = np.sum(freqs)

  # prepare corresponding word counts for genres that has the word
  matching_doc_word_sums = pd.concat([elem[1] for elem in df_word_sum_info]).as_matrix()
  # term-count inverse-document-frequency
  tcidf_values = transformer.fit_transform(freqs).toarray()
  # convert icidf values to tfidf values based on the formula in 
  # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
  return np.sum(tcidf_values / matching_doc_word_sums), word_count_sum
  # return np.sum(tcidf_values), word_count_sum

def custom_tokenizer(s):
  return s.split()

def flatten(elem, tu):
  return elem, tu[0], tu[1]

def docs(toplevel_file, limit=9000):

  file_path = TRAINING_DIR + toplevel_file
  text_filenames = [f for f in listdir(file_path) if isfile(join(file_path, f)) and (not f.startswith("."))][:limit]
  contents = [" ".join([stemmer.stem_word(w.lower()) for w in re.findall(r'[^0-9\W_]+', path.path(join(file_path, file_name)).text(errors="replace")) if len(w) < MAX_FEATURE_LENGTH]) for file_name in text_filenames]
  return text_filenames, contents

if __name__ == "__main__":
  tmps = []
  
  feature_list = set()

  for i, toplevel_file in enumerate(GENRE_FILENAMES):
    text_filenames, contents = docs(toplevel_file)

    print "line 62"

    vectorizer = CountVectorizer(stop_words=STOP_WORDS, decode_error ="replace", tokenizer=custom_tokenizer)

    print "line 66"

    fitted = vectorizer.fit_transform(contents)

    print "line 70"
    
    feature_names = vectorizer.get_feature_names()
    tmp = pd.DataFrame(fitted.toarray(), columns = feature_names, index = text_filenames)
    feature_list = feature_list | set(feature_names)
    
    print "line 76"

    length = tmp.shape[0]

    tmp['tags'] = pd.Series(np.repeat(i,length), index = tmp.index)

    tmp.to_sparse(fill_value=0)

    print "tmp shape: " + str(tmp.shape)

    tmps.append(tmp)

    print "line 88"  

  transformer = TfidfTransformer()

  doc_word_sums = [df.sum(axis=1) for df in tmps]

  zero_filled_series = [pd.Series(np.repeat(0, tmp.shape[0]), index=tmp.index) for tmp in tmps]

  # filter features based on the bag of words for all four genres
  word_tfidf_wc = [flatten(word, tfidf(word, tmps, transformer, doc_word_sums, zero_filled_series)) for word in feature_list]
  word_tfidf_wc.sort(key=lambda elem:elem[1])
  word_tfidf_wc.reverse()

  # choose the first FILTERED_FEATURE_SET_SIZE features with the largest tf-idf sums
  selected_features_tfidf = [elem[0] for elem in word_tfidf_wc[0:FILTERED_FEATURE_SET_SIZE_TFIDF]]

  word_tfidf_wc.sort(key=lambda elem:elem[2])
  word_tfidf_wc.reverse()

  selected_features_wc = [elem[0] for elem in word_tfidf_wc[WC_START_AT_INDEX:FILTERED_FEATURE_SET_SIZE_WC + WC_START_AT_INDEX]]

  union_features = set(selected_features_tfidf) | set(selected_features_wc)

  # preselect dataframe columns per genre before merging
  tmps = [tmp[["tags"] + [col for col in tmp.columns if col in union_features]] for tmp in tmps]

  df = pd.concat(tmps)
  df.fillna(0,inplace=True) 

  df.to_csv(join(OUTPUT_DIR, "doc_feature_top_wc_below_5000.csv"))
