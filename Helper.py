import re, os
from datetime import timedelta

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import pandas

stop_words = None
blanks = ' \t\n'
sentence_end = '. '


def create_folder(folder_path: str):
  if not os.path.exists(folder_path):
    os.mkdir(folder_path)


def initialize():
  global stop_words
  stop_words = set(stopwords.words('english'))


def remove_pattern(text: str, pattern_regex: str, replace_str: str = '') -> str:
  return re.sub(pattern_regex, replace_str, text)


def remove_stop_words(text: str) -> str:
  global stop_words

  word_tokens = word_tokenize(text)
  filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
  return ' '.join(filtered_sentence)


def daterange(start_date, end_date):
  for n in range(int((end_date - start_date).days)):
    yield start_date + timedelta(n)


def add_column_csv(text: str, after: int, value: str) -> str:
  splitted = text.split(',')
  splitted.insert(after, value)
  final_text = ','.join(splitted)
  return final_text

def get_pandas_column_names(file_path):
  dataframe = pandas.read_csv(file_path)
  return dataframe.columns