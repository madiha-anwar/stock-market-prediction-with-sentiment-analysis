class Stock:
  def __init__(self, ticker: str, cash_tag: str) -> None:
    self.ticker = ticker
    self.cash_tag_search = cash_tag


class Sentiment:
  def __init__(self, text: str, value: int) -> None:
    self.text = text
    self.value = value


class TweetSentiment:
  def __init__(self, date: str, text: str, sentiment: int = 0) -> None:
    self.date = date
    self.text = text
    self.sentiment = sentiment
