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

FILTERED_FEATURE_SET_SIZE_VAR = 1000
FILTERED_FEATURE_SET_SIZE_WC = 1000

MAX_FEATURE_LENGTH = 20

GENRE_FILENAMES = ["Child_0", "History_1", "Religion_2", "Science_3"]

STOP_WORDS = ["a","able","about","across","after","all","almost","also","am","among","an","and","any","are","as","at","be","because","been","but","by","can","cannot","could","dear","did","do","does","either","else","ever","every","for","from","get","got","had","has","have","he","her","hers","him","his","how","however","i","if","in","into","is","it","its","just","least","let","like","likely","may","me","might","most","must","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","she","should","since","so","some","than","that","the","their","them","then","there","these","they","this","tis","to","too","twas","us","wants","was","we","were","what","when","where","which","while","who","whom","why","will","with","would","yet","you","your",
              "project", "gutenberg", "ebook", "title", "author", "release", "chapter"]

stemmer = PorterStemmer()

def wc_appear(word, dfs, zero_filled_series):
  df_word_sum_info = [df.loc[:,word] if (word in df.columns) else zeros for df, zeros in zip(dfs, zero_filled_series)]

  counts = pd.concat(df_word_sum_info)
  print len(counts.nonzero()[0])
  return len(counts.nonzero()[0])


def wc(word, dfs, zero_filled_series):
  """
  Given a word, compute the sum of tf-idf values across all observations in the four genres
  """
  df_word_sum_info = [df.loc[:,word] if (word in df.columns) else zeros for df, zeros in zip(dfs, zero_filled_series)]

  counts = pd.concat(df_word_sum_info)

  return np.sum(counts)

def custom_tokenizer(s):
  return s.split()

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

  # filter features based on the bag of words for all four genres
  zero_filled_series = [pd.Series(np.repeat(0, tmp.shape[0]), index=tmp.index) for tmp in tmps]
  word_wc = [(word, wc(word, tmps, zero_filled_series)) for word in feature_list]
  word_wc.sort(key=lambda elem:elem[1])
  word_wc.reverse()

  word_wc_appears=[(word,wc_appear(word,tmps,zero_filled_series)) for word in feature_list]
  word_wc.sort(key=lambda elem:elem[1])
  word_wc.reverse()


  s1_total_wc = tmps[0].drop('tags', 1).sum(axis=1)
  s2_total_wc = tmps[1].drop('tags', 1).sum(axis=1)
  s3_total_wc = tmps[2].drop('tags', 1).sum(axis=1)
  s4_total_wc = tmps[3].drop('tags', 1).sum(axis=1)

  hist_col = tmps[1].drop('tags', 1).sum(axis=0)
  religion_col = tmps[2].drop('tags', 1).sum(axis=0)

  hist_word_wc = [(word, count) for word, count in zip(hist_col.index.values,hist_col.as_matrix())]
  religion_word_wc = [(word, count) for word, count in zip(religion_col.index.values,religion_col.as_matrix())]

  hist_word_wc.sort(key=lambda elem:elem[1])
  hist_word_wc.reverse()

  religion_word_wc.sort(key=lambda elem:elem[1])
  religion_word_wc.reverse()

  total_wc_vector = pd.concat([s1_total_wc, s2_total_wc, s3_total_wc, s4_total_wc])
  
  filtered_word_wc = [elem for elem in word_wc if (elem[1] > 50 and elem[1] < 5000)]
  filtered_word_wc.sort(key=lambda elem:elem[1])
  filtered_word_wc.reverse()

  filtered_word_wc_relaxed = [elem for elem in word_wc if (elem[1] > 2000 and elem[1] < 10000)]
  word_wc_hash = {word: wc for word, wc in filtered_word_wc_relaxed}

  hist_rank = [(word, wc * 1.0 / word_wc_hash[word]) for word, wc in hist_word_wc if word in word_wc_hash.keys()]
  hist_rank.sort(key=lambda elem:elem[1])
  hist_rank.reverse()

  religion_rank = [(word, wc * 1.0 / word_wc_hash[word]) for word, wc in religion_word_wc if word in word_wc_hash.keys()]
  religion_rank.sort(key=lambda elem:elem[1])
  religion_rank.reverse()

  union_features = set([elem[0] for elem in filtered_word_wc[0:2300]]) | set([elem[0] for elem in hist_rank[0:380]]) | set([elem[0] for elem in religion_rank[0:380]])
  
  df = pd.concat([tmp[["tags"] + [col for col in tmp.columns if col in union_features]] for tmp in tmps])
  df.fillna(0,inplace=True) 
  
  df.to_csv(join(OUTPUT_DIR, "doc_feature_top_words_and_category_unique_words.csv"))
