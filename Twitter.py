from datetime import datetime, timedelta
import twint

twitter_datetime_format = '%Y-%m-%d %H:%M:%S'

class Twitter:
  output_date_format: str = '%Y-%m-%d'

  @staticmethod
  def ScrapeTweets(cash_tag:str, from_date:str, to_date:str, output_path:str):
    # Configure
    c = twint.Config()
    # c.Limit = 10
    c.Show_cashtags = True
    # c.Stats = True
    c.Search = cash_tag
    c.Since = from_date
    c.Until = to_date
    c.Store_json = True
    c.Output = output_path

    # Run
    twint.run.Search(c)

class Tweet:
  def __init__(self, id: int, conversation_id: str, created_at: str, timezone: str, user_id: int, username: str, name: str, place: str, tweet: str, hashtags: list, cashtags: list) -> None:
    self.id = id
    self.coversation_id = conversation_id
    self.created_at = created_at
    self.date = ''
    self.time = ''
    self.timezone = timezone
    self.username = username
    self.tweet = tweet
    self.cleaned_tweet = tweet
    self.hashtags = hashtags
    self.cashtags = cashtags

  @property
  def utc_date(self):
    delta_hours = int(self.timezone[1:3])
    delta_minutes = int(self.timezone[-2:])
    # invert hours and minutes if timezone is negative
    if self.timezone[0] == '+':
      delta_hours   *= -1
      delta_minutes *= -1

    # removing timezone characters
    temp_created_at = ' '.join(self.created_at.split(' ')[:-1])
    temp_date = datetime.strptime(temp_created_at, twitter_datetime_format)
    # timezone conversion to utc
    return (temp_date + timedelta(hours=delta_hours, minutes=delta_minutes)).strftime(Twitter.output_date_format)
