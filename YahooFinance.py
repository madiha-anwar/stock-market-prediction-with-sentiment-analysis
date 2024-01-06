from pandas.core.frame import DataFrame
from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
yf.pdr_override()
import pandas as pd
import numpy as np


# this will be used to store yahoo finance dataset after cleanup columns 
class HistoricalData:
  def __init__(self, csv_text) -> None:
    columns = csv_text.split(',')
    self.date = columns[0]
    self.open = columns[1]
    self.close = columns[4]


class YahooFinance:
  opening_col = 'Open'
  closing_col = 'Close'
  low_col = 'Low'
  high_col = 'High'
  trend_col = 'Trend'
  future_trend_col = 'FutureTrend'
  ef1_col = 'EF_1'      # max-min scale of closing stock
  ef2_col = 'EF_2'      # fluctuation percentage of closing stock
  trend_values = ['Positive', 'Negative', 'Neutral']

  def __init__(self) -> None:
    pass
  
  @staticmethod
  def fetch_one_historical_data_file(ticker: str, start_date: str, end_date: str, output_path: str) -> None:
    data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    # remove an unwanted column
    del data['Adj Close']
    data.to_csv(output_path)

  @staticmethod
  def fetch_historical_data(stocks: list, start_date: str, end_date: str, output_folder_path: str, output_file_extension: str) -> None:
    for stock in stocks:
      output_file_path = output_folder_path + stock.ticker + output_file_extension
      YahooFinance.fetch_one_historical_data_file(stock.ticker, start_date, end_date, output_file_path)
  
  @staticmethod
  def add_extra_features(stocks: list, future_day_to_predict: int, input_base_path: str, output_base_path: str, file_extension: str):
    temp_trend_col = 'temp_calc_trend'
    temp_future_trend_col = 'temp_calc_future_trend'
    
    # conditions column
    def make_conditions(df: DataFrame, col_name: str):
      return [ (df[col_name] > 0), (df[col_name] < 0), (df[col_name] == 0)]      

    # this function generates trend and FutureTrend features from yahoo finance data
    for stock in stocks:
      input_file_path = input_base_path + stock.ticker + file_extension
      output_file_path = output_base_path + stock.ticker + file_extension

      yahoo_input_df = pd.read_csv(input_file_path)

      # calculating first extended feature 
      yahoo_input_df[YahooFinance.ef1_col] = (yahoo_input_df[YahooFinance.closing_col] - yahoo_input_df[YahooFinance.low_col]) \
                                              / (yahoo_input_df[YahooFinance.high_col] - yahoo_input_df[YahooFinance.low_col])
                                        
      # calculating second extended feature
      yahoo_input_df[YahooFinance.ef2_col] = (yahoo_input_df[YahooFinance.closing_col] - yahoo_input_df[YahooFinance.opening_col]) / yahoo_input_df[YahooFinance.opening_col] * 100
      
      # calculating values for trend and FutureTrend
      yahoo_input_df[temp_trend_col] = yahoo_input_df[YahooFinance.closing_col] - yahoo_input_df[YahooFinance.opening_col]
      yahoo_input_df[temp_future_trend_col] = yahoo_input_df[YahooFinance.closing_col] - yahoo_input_df[YahooFinance.closing_col].shift(-1 * future_day_to_predict)

      # creating trends cols
      yahoo_input_df[YahooFinance.trend_col] = np.select(make_conditions(yahoo_input_df, temp_trend_col), YahooFinance.trend_values)
      yahoo_input_df[YahooFinance.future_trend_col] = np.select(make_conditions(yahoo_input_df, temp_future_trend_col), YahooFinance.trend_values)

      # deleting temporary cols
      del yahoo_input_df[temp_trend_col]
      del yahoo_input_df[temp_future_trend_col]

      # removing last rows which can't have future trend
      yahoo_input_df = yahoo_input_df[: future_day_to_predict * (-1)]
      
      # saving new dataframe as csv
      yahoo_input_df.to_csv(output_file_path, index=False)
      
  def add_commodities_trends(stocks: list, commodities: list, input_base_path: str, output_base_path: str, file_extension: str):
    comparisonColumn = 'Date'
    
    commoditiesTrend = {}
    for stock in commodities:
      input_file_path = input_base_path + stock.ticker + file_extension
      commoditiesTrend[stock.ticker] = pd.read_csv(input_file_path)
      commoditiesTrend[stock.ticker].set_index(comparisonColumn, inplace=True)
    
    for stock in stocks:
      input_file_path = input_base_path + stock.ticker + file_extension
      output_file_path = output_base_path + stock.ticker + file_extension

      yahoo_input_df = pd.read_csv(input_file_path)
      
      for commodity in commodities:
        commodity_df = commoditiesTrend[commodity.ticker]
        columns_count = len(yahoo_input_df.columns)
        new_column_name = commodity.cash_tag_search.strip('$') + ' ' + YahooFinance.future_trend_col
        # commoditiesTrend[commodity.ticker][YahooFinance.future_trend_col]
        new_column_values = []

        for index, row in yahoo_input_df.iterrows():
          try:
            temp_selected_row = commodity_df.loc[row[comparisonColumn]]
            new_column_values.append(temp_selected_row[YahooFinance.future_trend_col])
          except KeyError:
            new_column_values.append('')

        yahoo_input_df.insert(columns_count - 2, new_column_name, new_column_values)
      # saving new dataframe as csv
      yahoo_input_df.to_csv(output_file_path, index=False)