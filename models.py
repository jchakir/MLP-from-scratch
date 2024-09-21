import math
from typing import NoReturn, Union, Iterator, Any, cast
import json

from tqdm import tqdm
import numpy as np

from layers import Layer, DenseLayer
from loses import Loss
from utilities import History, optimizer__type, loss_function__type



class Model:
  @staticmethod
  def __split_validation_data(
        X: np.ndarray, y: np.ndarray, val_split: float
      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert 0 < val_split < 1, Exception(f'{val_split} validation split not valid, must be between 0 and 1')
    data_size: int = X.shape[0]
    train_size: int = int(data_size * (1 - val_split)) 
    return X[: train_size], y[: train_size], X[train_size: ], y[train_size: ]

  @staticmethod
  def __batches(
        X: np.ndarray, y: np.ndarray, batch_size: int
      ) -> Iterator[tuple[int, np.ndarray, np.ndarray, int]]:
    data_size = X.shape[0]
    if y.ndim == 1:
      y = y.reshape(-1, 1)
    id = start = 0
    while start < data_size:
      end = min(data_size, start + batch_size)
      yield id, X[start: end], y[start: end], end - start
      id += 1
      start = end

  def __init__(self, layers: list[Layer]=None) -> None: # type: ignore
    self.__layers: list[DenseLayer]
    self.__loss  : Loss
    if layers is not None:
      self.createNetwork(layers)

  def createNetwork(self, layers: list[Layer]) -> Union[None, NoReturn]:
    if not bool(layers):
      raise Exception('Expected non empty layers')
    if len(layers) < 3:
      raise Exception('Minimun layers lenght is 3 including InputLayer')
    input_layer, *dense_layers = layers
    for layer in dense_layers:
      layer(input_layer)
      input_layer = layer
    self.__layers = cast(list[DenseLayer], dense_layers)

  def __fit_init_layers(self, optimizer: optimizer__type, lr: float, momentum_v: float, momentum_s: float) -> None:
    for layer in self.__layers:
      layer.init(optimizer, lr, momentum_v, momentum_s)

  def __early_stopping(self, loss: float, prev_loss: float, patience: int, tracker: int) -> tuple[str, int]:
    if loss < prev_loss:
      for layer in self.__layers: layer.backup()
      return 'continue', patience
    elif tracker <= 0:
      for layer in self.__layers: layer.restore()
      return 'stopping', patience
    else:
      return 'continue', tracker - 1

  def __foreward(self, X: np.ndarray) -> np.ndarray:
    for layer in self.__layers:
      X = layer.foreward(X)
    return X

  def __backward(self, gradient: np.ndarray) -> None:
    last: bool = True
    for layer in reversed(self.__layers):
      gradient = layer.backward(gradient, last=last)
      last = False

  def save(self, model_name: str='./model', *, verbose: bool=True) -> None:
    metadata = {
        'model': {'loss': self.__loss.loss , 'size': len(self.__layers)},
        'layers': {}
      }
    params: dict[str, Any] = {}
    for i, layer in enumerate(self.__layers):
      data = layer.save()
      params[f'w{i}'] = data.weights
      params[f'b{i}'] = data.bias
      metadata['layers'][f'{i}'] = data.output, data.activation
    with open(f'{model_name}.metadata', 'w') as file:
      json.dump(metadata, file, indent=2)
      np.savez_compressed(model_name, **params)
    if verbose:
      print(f"> saving model '{model_name}.npy, {model_name}.metadata' to disk...")

  def load(self, model_name: str) -> None:
    with open(f'{model_name}.metadata', 'r') as file:
      metadata = json.load(file)
      params = np.load(f'{model_name}.npz')

    loss, size = metadata['model']['loss'], metadata['model']['size']
    self.__layers = []
    for i in range(size):
      weights, bias      = params[f'w{i}'], params[f'b{i}']
      output, activation = metadata['layers'][f'{i}']
      layer = DenseLayer(output_shape=output, activation=activation)
      layer.load(weights=weights, bias=bias)
      self.__layers.append(layer)
    self.__loss = Loss(loss=loss)

  def test(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    match self.__loss.loss:
      case 'mse':
        raise NotImplementedError('Model.test.loss == mse not implemented yet')
      case 'binarycrossentropy':
        yHat     = self.__foreward(X)
        loss     = self.__loss.loss_value(y, yHat)
        accuracy, recall, precision, f1_score = self.__loss.metric(y, yHat)
        return {'loss':loss, 'accuracy':accuracy, 'recall':recall, 'precision':precision, 'f1_score':f1_score}
      case 'crossentropy':
        raise NotImplementedError('Model.test.loss == crossentropy not implemented yet')

  def predict(self, X: np.ndarray) -> np.ndarray:
    return self.__foreward(X)

  def fit(
      self, X: np.ndarray, y: np.ndarray, *, loss: loss_function__type,
      lr=0.0271, optimizer: optimizer__type='rmsprop', batch_size=100, epochs=13,
      X_val: np.ndarray=None, y_val: np.ndarray=None, val_split=0.2, # type: ignore
      momentum_v=0.9, momentum_s=0.99, early_stop=True, early_stop_patience=1
      ) -> History:
    """
    Model function responsible of train/fiting
    """
    # if validation data not provided, do split data into train and validation
    if X_val is None or y_val is None:
      X, y, X_val, y_val = self.__split_validation_data(X, y, val_split)

    history = History()
    
    self.__loss = Loss(loss=loss)

    self.__fit_init_layers(optimizer, lr, momentum_v, momentum_s)

    # ----------------------- fit helpers ------------------------
    def __fit_loss_and_matric_history(y: np.ndarray, yHat: np.ndarray) -> tuple[float, float, float, float]:
      yHat_val = self.__foreward(X_val)
      val_loss = self.__loss.loss_value(y_val, yHat_val)
      train_loss = self.__loss.loss_value(y, yHat)
      history.append_loss(loss=train_loss, val_loss=val_loss)

      if loss == 'mse':
        raise NotImplementedError('__fit_loss_and_matric_history() loss == mse not implemented yet')
      else:
        accuracy, recall, precision, f1_score = self.__loss.metric(y, yHat)
        val_accuracy, val_recall, val_precision, val_f1_score = self.__loss.metric(y_val, yHat_val)
        history.metric(accuracy=accuracy, recall=recall, precision=precision, f1_score=f1_score)
        history.val_metric(val_accuracy=val_accuracy, val_recall=val_recall, val_precision=val_precision, val_f1_score=val_f1_score)

      return train_loss, val_loss, accuracy, val_accuracy
    # -----------------------------------------------------------

    with tqdm(range(epochs), unit=' epoch') as tepoch:
      batch_count           : int = math.ceil(X.shape[0] / batch_size)
      prev_loss             : float = np.inf
      postfix_early_stop_str: str = ''
      tracker               : int = early_stop_patience
      for i in tepoch:
        for id, Xb, yb, size in self.__batches(X, y, batch_size):
          yHat = self.__foreward(Xb)
          gradient = self.__loss.derivative(yb, yHat, size)
          self.__backward(gradient)

          train_loss, val_loss, accuracy, val_accuracy = __fit_loss_and_matric_history(yb, yHat)
          if early_stop:
            early_stop_status, tracker = self.__early_stopping(val_loss, prev_loss, early_stop_patience, tracker)
            postfix_early_stop_str = f' early_stop={early_stop_status}'
            prev_loss = val_loss # assign val_loss to prev_loss

        postfix_loss_str = f'loss={train_loss:.3f} val_loss={val_loss:.3f}'
        postfix_accuracy_str = f'accuracy={accuracy:.3f} val_accuracy={val_accuracy:.3f}'
        tepoch.set_postfix_str(f'{postfix_loss_str} {postfix_accuracy_str}{postfix_early_stop_str}')
        tepoch.set_description(f'epoch {i+1: 3}/{epochs}, batch {id+1: 3}/{batch_count}')
    # return value
    return history
