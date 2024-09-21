import sys

import numpy as np
import pandas as pd

from models import Model
from utilities import layer_standardization



def test(model_name: str, X: np.ndarray, y: np.ndarray) -> None:
  model = Model()
  model.load(model_name)
  loss_and_metrics = model.test(X, y)
  print(loss_and_metrics)


def main() -> None:
  if len(sys.argv) != 5 or sys.argv[1] != '--dataset' or sys.argv[3] != '--model':
    print('invalid program arguments:')
    print(f'use: {sys.argv[0]} --dataset [train_dataset.csv] --model [model_name]')
    print('[train_dataset.csv]: Required, indicate the dataset file to train on.')
    print('[model_name]: Required, indicate the model files (.npz, .metadata) without extention files.')
    sys.exit(1)
  try:
    data = pd.read_csv(sys.argv[2])
    y = data['1'].to_numpy()
    X = data.drop(columns='1').to_numpy()
    X = layer_standardization(X)
    test(sys.argv[4], X, y)

  except (FileNotFoundError, PermissionError):
    print(f'File Not Found or Permission Error.')
    print(error)
  except NotImplementedError as error:
    print('NotImplementedError')
    print(error)
  except Exception as error:
    print('Exception')
    print(error)
  except:
    print('Internal Server Error.')
    print(error)


if __name__ == '__main__':
  main()
