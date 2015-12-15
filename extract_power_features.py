from os import listdir
from os.path import isfile, join
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tag import pos_tag, map_tag
import path
import pandas as pd
import numpy as np
import re
import pdb
import nltk
from collections import Counter


"""
Set these variables to be the top level directory of training samples (i.e. your_own_path/Training/) as well as the directory to store the result CSVs
"""
TRAINING_FEATURE_MATRIX = "/Users/fenglin/Desktop/stat154/Training_Output/doc_feature_top_words_and_category_unique_words.csv"
OUTPUT_DIR = "/Users/fenglin/Desktop/stat154/Training_Output/"
TRAINING_DIR = "/Users/fenglin/Desktop/stat154/Training/"

GENRE_FILENAMES = ["Child_0", "History_1", "Religion_2", "Science_3"]

STOP_WORDS = ["a","able","about","across","after","all","almost","also","am","among","an","and","any","are","as","at","be","because","been","but","by","can","cannot","could","dear","did","do","does","either","else","ever","every","for","from","get","got","had","has","have","he","her","hers","him","his","how","however","i","if","in","into","is","it","its","just","least","let","like","likely","may","me","might","most","must","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","she","should","since","so","some","than","that","the","their","them","then","there","these","they","this","tis","to","too","twas","us","wants","was","we","were","what","when","where","which","while","who","whom","why","will","with","would","yet","you","your",
              "project", "gutenberg", "ebook", "title", "author", "release", "chapter"]

# tag_set = ["p_ADJ","p_ADP","p_ADV","p_CONJ","p_DET","p_NOUN","p_NUM","p_PROM","p_PRT","p_VERB","p_X"]

power_info = pd.DataFrame(columns=["p_mean_sen_len", "p_sd_sen_len", "p_mean_word_len", "p_semi_colons_per_word","p_exclamation_per_word","p_quotations_per_word"]) #+ tag_set)

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
  for file_name in text_filenames:
    content_str = path.path(join(file_path, file_name)).text(errors="replace")
    sentences = nltk.sent_tokenize(content_str)

    print file_name

    # ct = Counter()

    # for sent in sentences:
    #   tokenized = nltk.word_tokenize(sent)
    #   tags = [map_tag('en-ptb', 'universal', tag) for word, tag in nltk.pos_tag(tokenized)]
    #   ct.update(tags)

    word_lengths = [len(w) for w in re.findall(r'[^\W_]+', content_str)]
    wc = len(word_lengths)
    sentence_lengths = [np.sum([len(w) for w in re.findall(r'[^\W_]+', sen)]) for sen in sentences]
    
    mean_sentence_len =  np.sum(sentence_lengths) / len(sentences)
    
    sd_sentence_len = np.std(sentence_lengths)

    mean_word_length = np.mean(word_lengths)
  
    semi_colons_per_word = content_str.count(",") * 1.0 / wc

    exclamation_per_word = content_str.count("!") * 1.0 / wc

    quotations_per_word = len(re.findall(r'\"|\'', content_str)) * 1.0 / wc

    # tag_res = [ct[k] * 1.0 / wc for k in tag_set]

    power_info.loc[file_name] = [mean_sentence_len, sd_sentence_len, mean_word_length, semi_colons_per_word, exclamation_per_word, quotations_per_word] #+ tag_res

if __name__ == "__main__":

  for i, toplevel_file in enumerate(GENRE_FILENAMES):

    docs(toplevel_file)

  training_pd = pd.DataFrame.from_csv(TRAINING_FEATURE_MATRIX)
  new_pd = pd.concat([training_pd, power_info], axis=1)
  new_pd.to_csv(join(OUTPUT_DIR, "with_power_features.csv"))
