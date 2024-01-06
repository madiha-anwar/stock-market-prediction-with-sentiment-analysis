import os

import SentimentAnalysis
from YahooFinance import YahooFinance
from Models import Stock
from MLAlgorithms.Algorithms import Algorithms
import Helper


if __name__ == '__main__':
  current_file_dir = os.path.dirname(__file__) + '/'
  data_folder = current_file_dir + 'data/'
  yahoo_finance_folder = data_folder + 'yahoo/'
  yahoo_finance_extended_folder = yahoo_finance_folder + 'extended-features/'

  prediction_folder = data_folder + 'prediction/'
  sentiment_prediction_folder = prediction_folder + 'sentiment/'
  final_prediction_folder = prediction_folder + 'final/'
  # folder to store predicted results of different classifiers
  ml_predicted_folder = data_folder + 'pridected/'

  Helper.create_folder(data_folder)
  Helper.create_folder(yahoo_finance_folder)
  Helper.create_folder(prediction_folder)
  Helper.create_folder(sentiment_prediction_folder)
  Helper.create_folder(final_prediction_folder)
  Helper.create_folder(yahoo_finance_extended_folder)
  Helper.create_folder(ml_predicted_folder)


  # for which day in future the FutureTrend is to be predicted
  days_in_future = 1

  start_date = '2018-07-01'
  # start_date = '2020-07-01'
  end_date = '2021-07-01'

  companies = [Stock('LSEG.L', '$LSE OR $LSEG'), Stock('HPQ', '$HPQ'), Stock('IBM', '$IBM')
            , Stock('ORCL', '$ORCL'), Stock('RHT.V', '$RHT OR $RHT.V'), Stock('TWTR', '$TWTR')
            , Stock('MSFT', '$MSFT')]
  commodities = [Stock('CL=F', 'Oil'), Stock('GC=F', 'Gold')]

  companies = [Stock('IBM', '$IBM')]

  # YahooFinance.fetch_historical_data(stocks=companies, start_date=start_date, end_date=end_date
  #               , output_folder_path=yahoo_finance_folder, output_file_extension='.csv')

  # YahooFinance.add_extra_features(stocks=companies, future_day_to_predict=days_in_future, input_base_path=yahoo_finance_folder, output_base_path=yahoo_finance_extended_folder, file_extension='.csv')
  
  # YahooFinance.fetch_historical_data(stocks=commodities, start_date=start_date, end_date=end_date
  #               , output_folder_path=yahoo_finance_folder, output_file_extension='.csv')
  # YahooFinance.add_extra_features(stocks=commodities, future_day_to_predict=days_in_future, input_base_path=yahoo_finance_folder, output_base_path=yahoo_finance_extended_folder, file_extension='.csv')

  # YahooFinance.add_commodities_trends(stocks=companies, commodities=commodities, input_base_path=yahoo_finance_extended_folder, output_base_path=yahoo_finance_extended_folder, file_extension='.csv')

  # SentimentAnalysis.PerformAnalysis(data_folder, start_date, end_date, companies, yahoo_finance_folder=yahoo_finance_extended_folder, prediction_data_folder=sentiment_prediction_folder)

  Algorithms.run_predictions(stocks=companies, input_file_extension='.csv', input_base_path=final_prediction_folder, output_base_path=ml_predicted_folder)

  exit()