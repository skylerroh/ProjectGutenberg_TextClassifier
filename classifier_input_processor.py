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
RAW_DATA_DIR = "/Users/fenglin/Desktop/stat154/Stat154FinalProject/Practice/"
OUTPUT_DIR = "/Users/fenglin/Desktop/stat154/Training_Output/"
TRAINING_FILE_PATH = "/Users/fenglin/Desktop/stat154/Training_Output/doc_feature_top_wc_top_tfidf_var.csv"
# TRAINING_FILE_PATH = "/Users/fenglin/Downloads/doc_feature_top_wc_below_5000.csv"
# FULL_FEATURE_NAMES_FILE = "/Users/fenglin/Desktop/stat154/Training_Output/full_feature_names.npy"
MAX_FEATURE_LENGTH = 20

GENRE_FILENAMES = ["Child(0)", "History(1)", "Religion(2)", "Science(3)"]

STOP_WORDS = ["a","able","about","across","after","all","almost","also","am","among","an","and","any","are","as","at","be","because","been","but","by","can","cannot","could","dear","did","do","does","either","else","ever","every","for","from","get","got","had","has","have","he","her","hers","him","his","how","however","i","if","in","into","is","it","its","just","least","let","like","likely","may","me","might","most","must","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","she","should","since","so","some","than","that","the","their","them","then","there","these","they","this","tis","to","too","twas","us","wants","was","we","were","what","when","where","which","while","who","whom","why","will","with","would","yet","you","your",
              "project", "gutenberg", "ebook", "title", "author", "release", "chapter"]

stemmer = PorterStemmer()

def custom_tokenizer(s):
  return s.split()

def docs(file_path, limit=9000):
  text_filenames = [f for f in listdir(file_path) if isfile(join(file_path, f)) and (not f.startswith("."))][:limit]
  contents = [" ".join([stemmer.stem_word(w.lower()) for w in re.findall(r'[^0-9\W_]+', path.path(join(file_path, file_name)).text(errors="replace")) if len(w) < MAX_FEATURE_LENGTH]) for file_name in text_filenames]
  return text_filenames, contents

if __name__ == "__main__":
  tmps = []
  
  feature_list = set()

  # full_feature_names = np.load(FULL_FEATURE_NAMES_FILE)
  training_df = pd.read_csv(TRAINING_FILE_PATH,header=0,index_col=0)
  training_df = training_df.drop('tags', 1)
  full_feature_names = training_df.columns.values

  text_filenames, contents = docs(RAW_DATA_DIR)


  vectorizer = CountVectorizer(stop_words=STOP_WORDS, decode_error ="replace", tokenizer=custom_tokenizer)
  fitted = vectorizer.fit_transform(contents)
  feature_names = vectorizer.get_feature_names()

  test_df = pd.DataFrame(fitted.toarray(), columns = feature_names, index = text_filenames)

  test_df_filtered = test_df[[col for col in test_df.columns if col in set(full_feature_names)]] 

  print test_df_filtered.shape

  empty_df = pd.DataFrame({}, columns=full_feature_names)

  output_df = pd.concat([test_df_filtered, empty_df])

  print output_df.shape

  output_df.fillna(0,inplace=True) 

  output_df.to_csv(join(OUTPUT_DIR, "test_feature_matrix_top_wc_top_tfidf_var.csv"))

