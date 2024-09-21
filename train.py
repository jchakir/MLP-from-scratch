import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import Model
import layers
from utilities import History, layer_standardization



def train(X: np.ndarray, y: np.ndarray) -> History:
  model = Model([
    layers.InputLayer(input_features=X.shape[1]),
    layers.DenseLayer(13, activation='sigmoid', weights_initializer='he_uniform'),
    layers.DenseLayer(13, activation='sigmoid', weights_initializer='he_uniform'),
    layers.DenseLayer(13, activation='sigmoid', weights_initializer='he_uniform'),
    layers.DenseLayer(13, activation='sigmoid', weights_initializer='he_uniform'),
    layers.DenseLayer(1, activation='sigmoid', weights_initializer='he_uniform')
  ])
  history = model.fit(
    X, y, loss='binarycrossentropy',
    epochs=35, batch_size=50, optimizer='adam',
    early_stop=False, early_stop_patience=1
  )
  model.save()
  return history


def visualize_history(history: History) -> None:
  _, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 5))
  # Axis 1
  ax1.plot(history.loss, label='Train Loss', color='green')
  ax1.plot(history.val_loss, label='Validation Loss', color='orange')
  ax1.set_title('Training and Validation Loss')
  ax1.text(0.5, 0.04, 'Epoch', ha='center')
  # Axis 2
  # ax2.plot(history.accuracy, label='Train Acc', color='green')
  # ax2.plot(history.val_accuracy, label='Validation Acc', color='orange')
  # ax2.set_title('Training and Validation Accuracy')
  # ax2.text(0.5, 0.04, 'Epoch', ha='center')
  # PLT Setting
  plt.legend()
  plt.tight_layout()
  plt.show()


def main() -> None:
  if len(sys.argv) != 3 or sys.argv[1] != '--dataset':
    print('invalid program arguments:')
    print(f'use: {sys.argv[0]} --dataset [train_dataset.csv]')
    print('[train_dataset.csv]: Required, indicate the dataset file to train on.')
    sys.exit(1)
  try:
    data = pd.read_csv(sys.argv[2])
    y = data['1'].to_numpy()
    X = data.drop(columns='1').to_numpy()
    X = layer_standardization(X)
    history = train(X, y)
    visualize_history(history)

  except (FileNotFoundError, PermissionError):
    print(f'File Not Found or Permission Error of [{sys.argv[2]}] dataset file')
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
