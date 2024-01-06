from StanfordNLP import StanfordNLP
from datetime import date, datetime, timedelta
import json
import os
import re

from Twitter import Tweet, Twitter
from Models import TweetSentiment
import Helper


def scrape_tweets(base_path: str, stocks: list, start_date: str, end_date: str) -> None:
  date_format = '%Y-%m-%d'
  start_date_obj = datetime.strptime(start_date, date_format)
  end_date_obj = datetime.strptime(end_date, date_format)

  for stock in stocks:
    # iterate over each month from start date to end date
    output_file_path = base_path + stock.ticker + '.json'
    for scrape_start_date in Helper.daterange(start_date=start_date_obj, end_date=end_date_obj):
      scrape_end_date = scrape_start_date + timedelta(days=1)
      print('Scraping data for:\t', output_file_path, "\nDate:\t", scrape_start_date.strftime(date_format))
      Twitter.ScrapeTweets(cash_tag=stock.cash_tag_search, from_date=scrape_start_date.strftime(date_format), to_date=scrape_end_date.strftime(date_format), output_path=output_file_path)


def clean_tweets(output_base_path: str, input_base_path: str, filename: str, input_file_extension: str, output_file_extension: str) -> None:
  # get only required data from tweets
  tweets = list()
  unique_tweet_dict = dict()
  input_file_path = input_base_path + filename + input_file_extension

  print('Cleaning tweets for file:\t', input_file_path)
  with open(input_file_path) as raw_tweets_file:    
    for tweet in raw_tweets_file:
      if tweet.strip('\n\t ') != '':
        try:
          tweet_json_obj = json.loads(tweet)
        except Exception as ex:
          print(tweet)
          print(ex)
          exit()

        tweet_obj = Tweet(id=tweet_json_obj['id'], conversation_id=tweet_json_obj['conversation_id'], created_at=tweet_json_obj['created_at']
                      , timezone=tweet_json_obj['timezone'], user_id=tweet_json_obj['user_id'], username=tweet_json_obj['username']
                      , name=tweet_json_obj['name'], place=tweet_json_obj['place'], tweet=tweet_json_obj['tweet']
                      , hashtags=tweet_json_obj['hashtags'], cashtags=tweet_json_obj['cashtags'])

        # remove URL
        tweet_obj.cleaned_tweet = Helper.remove_pattern(tweet_obj.cleaned_tweet, r'http\S+\s*')
        # remove retweets(RT) or cc 
        tweet_obj.cleaned_tweet = Helper.remove_pattern(tweet_obj.cleaned_tweet, r'RT|cc')
        # removing @tags
        tweet_obj.cleaned_tweet = Helper.remove_pattern(tweet_obj.cleaned_tweet, r'@\S+')
        # removing $cashtags
        tweet_obj.cleaned_tweet = Helper.remove_pattern(tweet_obj.cleaned_tweet, r'\$\S+')
        # removing #hashtags
        tweet_obj.cleaned_tweet = Helper.remove_pattern(tweet_obj.cleaned_tweet, r'#\S+')
        # removing &emp; or html quoted characters
        tweet_obj.cleaned_tweet = Helper.remove_pattern(tweet_obj.cleaned_tweet, r'&\w+;')
        # remove punctuations
        tweet_obj.cleaned_tweet = Helper.remove_pattern(tweet_obj.cleaned_tweet, r'[%s]' % re.escape("""!"#$%&'()*+,–-./:;<=>?@[\]^_`{|}~’"""))
        # remove words with only digits
        tweet_obj.cleaned_tweet = Helper.remove_pattern(tweet_obj.cleaned_tweet, r'\b\d+')
        # remove extra stop words
        tweet_obj.cleaned_tweet = Helper.remove_stop_words(tweet_obj.cleaned_tweet)
        # remove any non english letters
        tweet_obj.cleaned_tweet = Helper.remove_pattern(tweet_obj.cleaned_tweet, r'[^\w \t\n]+')
        
        # remove extra white spaces on the end
        tweet_obj.cleaned_tweet = tweet_obj.cleaned_tweet.strip()

        # add only non empty unique tweets
        if tweet_obj.cleaned_tweet != '' and tweet_obj.cleaned_tweet not in unique_tweet_dict:
          tweets.append(tweet_obj)
          unique_tweet_dict[tweet_obj.cleaned_tweet] = True

  cleaned_file_path = output_base_path + filename + output_file_extension
  # write cleaned tweets in the file
  with open(cleaned_file_path, 'w+') as cleaned_tweets_file:
    for tweet in tweets:
      csv_line = ','.join([str(tweet.id), tweet.utc_date, tweet.cleaned_tweet])
      cleaned_tweets_file.write(csv_line + '\n')
  return None


def clean_tweet_files(output_base_path: str, input_base_path: str, filenames: list
                      , file_extension: str, output_file_extension: str) -> None:
  for filename in filenames:
    clean_tweets(output_base_path, input_base_path, filename=filename, input_file_extension=file_extension
                , output_file_extension=output_file_extension)
  return None


def PerformAnalysis(data_folder: str, start_date: str, end_date: str, stocks: list
                    , yahoo_finance_folder: str, prediction_data_folder: str) -> None:
  tweets_directory = data_folder + 'tweets/'
  raw_tweets_directory = tweets_directory + 'raw/'
  cleaned_tweets_directory = tweets_directory + 'cleaned/'

  # remove tweets folder
  # shutil.rmtree(tweets_directory)

  if not os.path.exists(tweets_directory):
    os.mkdir(tweets_directory)
  if not os.path.exists(raw_tweets_directory):
    os.mkdir(raw_tweets_directory)
  if not os.path.exists(cleaned_tweets_directory):
    os.mkdir(cleaned_tweets_directory)

  raw_tweets_file_type = '.json'
  cleaned_tweets_file_type = '.csv'
  stocks_filenames = [stock.ticker for stock in stocks]

  Helper.initialize()
  # scrape_tweets(base_path=raw_tweets_directory, stocks=stocks, start_date=start_date, end_date=end_date)
  # clean_tweet_files(output_base_path=cleaned_tweets_directory, input_base_path=raw_tweets_directory
  #               , filenames=stocks_filenames, file_extension=raw_tweets_file_type
  #               , output_file_extension=cleaned_tweets_file_type)
  

  ################################ perform sentiment analysis 

  # sentiment column position to be second last column in output csv
  sentiment_column = -1
  sentiment_column_title = 'Twitter Sentiments'
  yahoo_date_column = 0   # first column is the date column

  for stock in stocks:
    # open cleaned tweets file and generate a dictionary with sentiments date wise
    date_sentiment_dict = dict()
    cleaned_tweets_dict = dict()
    complete_tweet_text = ''

    cleaned_tweets_file_path = cleaned_tweets_directory + stock.ticker + cleaned_tweets_file_type
    print('Starting sentiment analysis for:\t', cleaned_tweets_file_path)

    # read file and aggregate sentiments date wise
    with open(cleaned_tweets_file_path) as cleaned_file:
      for cleaned_tweet in cleaned_file:
        if cleaned_tweet.strip(Helper.blanks) != '':
          tweet_id, tweet_date, tweet_text = cleaned_tweet.split(',')   # it should have tweet id, date, tweet text
          cleaned_tweets_dict[tweet_text] = TweetSentiment(date=tweet_date, text=tweet_text)
          
          temp_next_len = len(complete_tweet_text) + len(tweet_text) + len(Helper.sentence_end)
          # if complete text is getting larger than max char limit then send the request and parse the response
          if temp_next_len > StanfordNLP.max_char_limit:
            # get sentiments of all tweets combined
            sentiments = StanfordNLP.get_sentiment(complete_tweet_text)
            for sentiment in sentiments:
              # removing sentence ending attached before
              tweet_original_text = sentiment.text.strip(Helper.sentence_end)

              if tweet_original_text not in cleaned_tweets_dict:
                print(tweet_original_text)
                print(sentiment.text)
                continue
              if cleaned_tweets_dict[tweet_original_text].date not in date_sentiment_dict:
                date_sentiment_dict[cleaned_tweets_dict[tweet_original_text].date] = 0
              # aggregating sentiment for the date
              date_sentiment_dict[cleaned_tweets_dict[tweet_original_text].date] += sentiment.value
              # emptying the aggregate tweet text for next iteration
              complete_tweet_text = ''

          complete_tweet_text += (tweet_text + Helper.sentence_end)      # appending sentence ending for a tweet

    # read yahoo finance file and add sentiment data column
    with open(yahoo_finance_folder + stock.ticker + cleaned_tweets_file_type) as yahoo_file:
      # output file to include sentiment data
      with open(prediction_data_folder + stock.ticker + cleaned_tweets_file_type, 'w+') as output_file:
        yahoo_file_lines = yahoo_file.read().split('\n')
        # copy header
        temp_text = Helper.add_column_csv(text=yahoo_file_lines[0], after=sentiment_column, value=sentiment_column_title)
        output_file.write(temp_text + '\n')

        for line_number in range(1, len(yahoo_file_lines)):
          if yahoo_file_lines[line_number].strip(Helper.blanks) != '':
            yahoo_date = yahoo_file_lines[line_number].split(',')[yahoo_date_column]
            # default sentiment value
            sentiment_value = 'NA'
            if yahoo_date in date_sentiment_dict:
              sentiment_value = str(date_sentiment_dict[yahoo_date])
            
            temp_text = Helper.add_column_csv(text=yahoo_file_lines[line_number], after=sentiment_column, value=sentiment_value)
            output_file.write(temp_text + '\n')          
  
  return None
