import numpy as np
from pycorenlp import StanfordCoreNLP

from Models import Sentiment


class StanfordNLP():
  max_char_limit = 99500
  nlp = StanfordCoreNLP('http://localhost:9000')
  
  def __init__(self) -> None:
    pass

  # Function; Output = # sentence, # words, avg.sentimentValue, sentimentHist
  @classmethod
  def get_sentiment(self, text_str) -> list:
    res = StanfordNLP.nlp.annotate(text_str,
                  properties={
                    'annotators': 'sentiment',
                    'outputFormat': 'json',
                    'timeout': 36000000,
                  })

    if isinstance(res, str):
      print('Unexpected response from Stanford NLP for text:\t', text_str)
      print(res)
      return (0, 0, 0, 0, [])

    sentiment_result = []

    for i, s in enumerate(res["sentences"]):
      original_sentence = ''.join([token['originalText'] + token['after'] for token in s['tokens']])
      sentiment_result.append(Sentiment(text=original_sentence, value=int(s['sentimentValue'])))

    return sentiment_result
