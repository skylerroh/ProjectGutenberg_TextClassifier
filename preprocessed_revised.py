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

"""
Set these variables to be the top level directory of training samples (i.e. your_own_path/Training/) as well as the directory to store the result CSVs
"""
TRAINING_DIR = "/Users/Skyroh/Desktop/sta154/Training/"
OUTPUT_DIR =  "/Users/Skyroh/Desktop/sta154/output"



GENRE_FILENAMES = ["Child(0)", "History(1)", "Religion(2)", "Science(3)"]

STOP_WORDS = ["a","able","about","across","after","all","almost","also","am","among","an","and","any","are","as","at","be","because","been","but","by","can","cannot","could","dear","did","do","does","either","else","ever","every","for","from","get","got","had","has","have","he","her","hers","him","his","how","however","i","if","in","into","is","it","its","just","least","let","like","likely","may","me","might","most","must","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","she","should","since","so","some","than","that","the","their","them","then","there","these","they","this","tis","to","too","twas","us","wants","was","we","were","what","when","where","which","while","who","whom","why","will","with","would","yet","you","your","project", "gutenberg", "ebook", "title", "author", "release", "chapter"]

MAX_FEATURE_LENGTH = 25

stemmer = PorterStemmer()

def custom_tokenizer(s):
  return s.split()

def docs(toplevel_file, limit=9000):
  """
  *** We need a way to open large dataset in R. A full matrix for the entire dataset will crash R for now.
  """
  file_path = TRAINING_DIR + toplevel_file
  text_filenames = [f for f in listdir(file_path) if isfile(join(file_path, f)) and (not f.startswith("."))][:limit]
  contents = [" ".join([stemmer.stem_word(w.lower()) for w in re.findall(r'[^\W_]+', path.path(join(file_path, file_name)).text(errors="replace")) if len(w) < MAX_FEATURE_LENGTH]) for file_name in text_filenames]
  return text_filenames, contents


if __name__ == "__main__":
  df = pd.DataFrame().to_sparse(fill_value=0)
  for i, toplevel_file in enumerate(GENRE_FILENAMES):
    text_filenames, contents = docs(toplevel_file)
    vectorizer = CountVectorizer(stop_words=STOP_WORDS, decode_error ="replace", tokenizer=custom_tokenizer)
    fitted = vectorizer.fit_transform(contents)
<<<<<<< HEAD
    tmp = pd.DataFrame(fitted.toarray(), columns = vectorizer.get_feature_names(), index = text_filenames)
    s=tmp.sum(axis=0)
    tmp = tmp.loc[:,s > 1]
    length = tmp.shape[0]
    tmp['tags'] = pd.Series(np.repeat(i,length), index = tmp.index)
=======
    # transformer = TfidfTransformer()
    # tfidf = transformer.fit_transform(fitted.toarray()).toarray()
    # x = np.argsort((tfidf.sum(axis=0)))[::-1]
    # index = x[1:1000]
    tmp = pd.DataFrame(fitted.toarray(), columns = vectorizer.get_feature_names(), index = text_filenames)
    # tmp = tmp.iloc[:,index]
    # length = tmp.shape[0]
    #s=tmp.sum(axis=0) 
    #tmp = tmp.loc[:,s > 15]
    # tmp['tags'] = pd.Series(np.repeat(i,length), index = tmp.index)
>>>>>>> 443a0f028f6a278f18f2fb2eb879e77e51956ad4
    df = pd.concat([tmp, df])


#s=tmp.sum(axis=0)
#tmp = tmp.loc[:,s > 15]
transformer = TfidfTransformer()
df = transformer.fit_transform(df).toarray()
x = np.argsort((df.sum(axis=0)))[::-1]
x = x[1:2000]
tmp = df.iloc[:,x]

  df.fillna(0,inplace=True) 

  #df.to_csv(join(OUTPUT_DIR, "WordFeature.csv"))

  # *** A feasible way to store large output. However, this is somehow not compatible with R.

  # hdf = pd.HDFStore(join(OUTPUT_DIR, "%s.h5" % (toplevel_file)))
  # hdf.put('d1',df,data_columns=True)
  # hdf.close()
