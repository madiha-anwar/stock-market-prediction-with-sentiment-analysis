from datetime import datetime
from MLAlgorithms.LSTM_Prediction import LSTM
from MLAlgorithms.CART_GBM_Tuning import CART_GBM_Tuning
from MLAlgorithms.Deep_Learning import Deep_learning
from MLAlgorithms.Voting_Ensemble import Voting_Ensemble


class Algorithms:
  outut_date_format = '%Y-%m-%d %H:%M:%S'

  def __init__(self) -> None:
    pass

  @classmethod
  def run_predictions(self, stocks: list, input_file_extension: str, input_base_path: str, output_base_path: str):
    current_run_folder = output_base_path + datetime.now().strftime(self.outut_date_format)

    for stock in stocks:
      input_file_path = input_base_path + stock.ticker + input_file_extension
      # CART_GBM_Tuning.run(dataset_file_path=input_file_path)
      # Deep_learning.run(dataset_file_path=input_file_path)

      # LSTM.run(dataset_file_path=input_file_path)

      Voting_Ensemble.run(dataset_file_path=input_file_path)
      break

