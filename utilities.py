from typing import NamedTuple, Literal, TypeAlias

import numpy as np



class History:
  def __init__(self) -> None:
    self.loss         : list[float] = []
    self.val_loss     : list[float] = []
    self.accuracy     : list[float] = []
    self.val_accuracy : list[float] = []
    self.recall       : list[float] = []
    self.val_recall   : list[float] = []
    self.precision    : list[float] = []
    self.val_precision: list[float] = []
    self.f1_score     : list[float] = []
    self.val_f1_score : list[float] = []

  def append_loss(self, *, loss=.0, val_loss=.0) -> None:
    self.loss.append(loss)
    self.val_loss.append(val_loss)

  def metric(self, *, accuracy=.0, recall=.0, precision=.0, f1_score=.0) -> None:
    self.accuracy.append(accuracy)
    self.recall.append(recall)
    self.precision.append(precision)
    self.f1_score.append(f1_score)

  def val_metric(self, *, val_accuracy=.0, val_recall=.0, val_precision=.0, val_f1_score=.0) -> None:
    self.val_accuracy.append(val_accuracy)
    self.val_recall.append(val_recall)
    self.val_precision.append(val_precision)
    self.val_f1_score.append(val_f1_score)


class DenseLayerData(NamedTuple):
  weights    : np.ndarray
  bias       : np.ndarray
  activation : str
  output     : int


def layer_standardization(X: np.ndarray) -> np.ndarray:
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    return (X - mean) / std

def batch_standardization(X: np.ndarray) -> np.ndarray:
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    return (X - mean) / std


weights_initializer__type: TypeAlias = Literal[
    'random', 'uniform', 'xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal'
  ]

optimizer__type: TypeAlias = Literal[
    'gd', 'gradient_descent', 'nag', 'nmgd', 'nesterov_momentum_gradient_descent', 'rmsprop', 'adam'
  ]

activation__type: TypeAlias = Literal[
    'linear', 'relu', 'sigmoid', 'softmax'
  ]

loss_function__type: TypeAlias = Literal[
    'binarycrossentropy', 'crossentropy', 'mse'
  ]
