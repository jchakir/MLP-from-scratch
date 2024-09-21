from typing import Callable

import numpy as np

from utilities import loss_function__type


class Loss:
  def __init__(self, loss: loss_function__type) -> None:
    loss_derivative: dict[str, Callable] = {
      'binarycrossentropy': self.__binary_cross_entropy_derivative,
      'crossentropy': self.__cross_entropy_derivative,
      'mse': self.__mse_derivative
    }
    loss_function: dict[str, Callable] = {
      'binarycrossentropy': self.__binary_cross_entropy_loss,
      'crossentropy': self.__cross_entropy_loss,
      'mse': self.__mse_loss
    }
    metrics: dict[str, Callable] = {
      'binarycrossentropy': self.__binary_cross_entropy_metric,
      'crossentropy': self.__cross_entropy_metric,
      'mse': self.__mse_metric
    }
    try:
      self.__loss: loss_function__type = loss
      self.__loss_derivative: Callable = loss_derivative[self.__loss]
      self.__loss_function: Callable = loss_function[self.__loss]
      self.__metric: Callable = metrics[self.__loss]
    except KeyError:
      raise NotImplementedError(f"Not Implemented Loss Function `{loss}'")

  def __mse_derivative(
        self, y: np.ndarray, yHat: np.ndarray, batch_size: int
      ) -> np.ndarray:
    return (yHat - y) / batch_size

  def __binary_cross_entropy_derivative(
        self, y: np.ndarray, yHat: np.ndarray, batch_size: int
      ) -> np.ndarray:
    return (yHat - y) / batch_size

  def __cross_entropy_derivative(
        self, y: np.ndarray, yHat: np.ndarray, batch_size: int
      ) -> np.ndarray:
    return (yHat - y) / batch_size

  def __mse_loss(self, y: np.ndarray, yHat: np.ndarray) -> float:
    loss = np.power(yHat - y, 2)
    return loss.mean()

  def __binary_cross_entropy_loss(self, y: np.ndarray, yHat: np.ndarray) -> float:
    yHat = np.clip(yHat, 1e-7, 1 - 1e-7)
    loss = y * np.log(yHat) + (1 - y) * np.log(1 - yHat)
    return -1 * loss.mean()

  def __cross_entropy_loss(self, y: np.ndarray, yHat: np.ndarray) -> float:
    yHat = np.clip(yHat, 1e-7, 1)
    loss = y * np.log(yHat)
    loss = np.sum(loss, axis=1)
    return -1 * loss.mean()

  def __mse_metric(self, y: np.ndarray, yHat: np.ndarray) -> tuple[float, float, float]:
    """
    Params
    ------
    y: true y, yHat: predicted y
    
    Returns
    -------
    mse, rmse, mae
    """
    y_yHat = y - yHat
    mse  = np.power(y_yHat, 2).mean()
    rmse = np.sqrt(mse)
    mae  = np.abs(y_yHat).mean()
    return mse, rmse, mae

  def __binary_cross_entropy_metric(
      self, y: np.ndarray, yHat: np.ndarray
      ) -> tuple[float, float, float, float]:
    """
    Params
    ------
    y: true y, yHat: predicted y
    
    Returns
    -------
    accuracy, recall, precision, f1_score
    """
    yHat = yHat > 0.5
    y1, y0, yHat1, yHat0 = y == 1, y == 0, yHat == 1, yHat == 0
    tp = np.sum((y1) & (yHat1))
    fn = np.sum((y1) & (yHat0))
    fp = np.sum((y0) & (yHat1))
    tn = np.sum((y0) & (yHat0))
    accuracy  = (tp + tn) / (tp + tn + fp + fn)
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score  = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, recall, precision, f1_score

  def __cross_entropy_metric(self, y: np.ndarray, yHat: np.ndarray) -> tuple[float]:
    raise NotImplementedError('Loss.__cross_entropy_metric() not implemented yet')

  @property
  def loss(self) -> loss_function__type:
    return self.__loss

  def derivative(self, y: np.ndarray, yHat: np.ndarray, batch_size: int) -> np.ndarray:
    return self.__loss_derivative(y, yHat, batch_size)

  def loss_value(self, y: np.ndarray, yHat: np.ndarray) -> float:
    return self.__loss_function(y, yHat)
  
  def metric(self, y: np.ndarray, yHat: np.ndarray) -> tuple:
    return self.__metric(y, yHat)
