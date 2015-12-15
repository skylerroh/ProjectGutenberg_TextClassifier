from os import listdir
from os.path import isfile, join
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import PorterStemmer
import path
import nltk
import pandas as pd
import numpy as np
import re
import pdb

"""
Set these variables to be the top level directory of training samples (i.e. your_own_path/Training/) as well as the directory to store the result CSVs
"""
RAW_DATA_DIR = "/Users/fenglin/Downloads/Validation/"
OUTPUT_DIR = "/Users/fenglin/Desktop/stat154/Training_Output/"
TRAINING_FILE_PATH = "/Users/fenglin/Desktop/stat154/Training_Output/doc_feature_top_words_and_category_unique_words.csv"
MAX_FEATURE_LENGTH = 20

GENRE_FILENAMES = ["Child_0", "History_1", "Religion_2", "Science_3"]

STOP_WORDS = ["a","able","about","across","after","all","almost","also","am","among","an","and","any","are","as","at","be","because","been","but","by","can","cannot","could","dear","did","do","does","either","else","ever","every","for","from","get","got","had","has","have","he","her","hers","him","his","how","however","i","if","in","into","is","it","its","just","least","let","like","likely","may","me","might","most","must","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","she","should","since","so","some","than","that","the","their","them","then","there","these","they","this","tis","to","too","twas","us","wants","was","we","were","what","when","where","which","while","who","whom","why","will","with","would","yet","you","your",
              "project", "gutenberg", "ebook", "title", "author", "release", "chapter"]

power_info = pd.DataFrame(columns=["p_mean_sen_len", "p_sd_sen_len", "p_mean_word_len", "p_semi_colons_per_word","p_exclamation_per_word","p_quotations_per_word"]) #+ tag_set)

stemmer = PorterStemmer()

def custom_tokenizer(s):
  return s.split()

def docs(file_path, limit=9000):
  text_filenames = [f for f in listdir(file_path) if isfile(join(file_path, f)) and (not f.startswith("."))][:limit]
  contents = [" ".join([stemmer.stem_word(w.lower()) for w in re.findall(r'[^0-9\W_]+', path.path(join(file_path, file_name)).text(errors="replace")) if len(w) < MAX_FEATURE_LENGTH]) for file_name in text_filenames]
  
  for file_name in text_filenames:

    content_str = path.path(join(file_path, file_name)).text(errors="replace")
    sentences = nltk.sent_tokenize(content_str)

    word_lengths = [len(w) for w in re.findall(r'[^\W_]+', content_str)]
    wc = len(word_lengths)
    sentence_lengths = [np.sum([len(w) for w in re.findall(r'[^\W_]+', sen)]) for sen in sentences]
    
    mean_sentence_len =  np.sum(sentence_lengths) / len(sentences)
    
    sd_sentence_len = np.std(sentence_lengths)

    mean_word_length = np.mean(word_lengths)
  
    semi_colons_per_word = content_str.count(",") * 1.0 / wc

    exclamation_per_word = content_str.count("!") * 1.0 / wc

    quotations_per_word = len(re.findall(r'\"|\'', content_str)) * 1.0 / wc

    power_info.loc[file_name] = [mean_sentence_len, sd_sentence_len, mean_word_length, semi_colons_per_word, exclamation_per_word, quotations_per_word] #+ tag_res


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

  print "test matrix before merging with predefined features " + str(test_df_filtered.shape)

  empty_df = pd.DataFrame({}, columns=full_feature_names)

  output_df = pd.concat([test_df_filtered, empty_df])

  print "output matrix shape: " + str(output_df.shape)

  output_df.fillna(0,inplace=True) 

  new_pd = pd.concat([output_df, power_info], axis=1)
  new_pd.index = [int(elem[:-4]) for elem in output_df.index]

  new_pd = new_pd.sort_index()

  new_pd.to_csv(join(OUTPUT_DIR, "test_matrix_with_power_features.csv"))
