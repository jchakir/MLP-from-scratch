from typing import Callable, Any
from abc import ABC

import numpy as np

from activations import Activation
from utilities import DenseLayerData, weights_initializer__type, optimizer__type, activation__type



class Layer(ABC):
  def __init__(self, *, type: str, output_shape: int) -> None:
    self._type: str = type
    self._output_shape: int = output_shape

  def __call__(self, *args: Any, **kwds: Any) -> Any:
    raise NotImplementedError(f"The Layer of type `{self._type}' is not callable.")

  @property
  def output_shape(self) -> int:
    return self._output_shape
  
  @property
  def type(self) -> str:
    return self._type


class InputLayer(Layer):
  def __init__(self, *, input_features: int) -> None:
    super().__init__(type='input', output_shape=input_features)


class DenseLayer(Layer):
  def __init__(
        self, output_shape: int, activation: activation__type,
        weights_initializer: weights_initializer__type = 'xavier_uniform'
      ) -> None:
    super().__init__(type='dense', output_shape=output_shape)
    self.__optimizer_foreward  : Callable = self.__gd_foreward
    self.__optimizer_backward  : Callable
    self.__weights_initializer : weights_initializer__type = weights_initializer
    self.__activation          : Activation = Activation(activation)
    self.__X                   : np.ndarray
    self.__weights             : np.ndarray
    self.__bias                : np.ndarray
    self.__weights_backup      : np.ndarray
    self.__bias_backup         : np.ndarray

  def __gd_init(self) -> None:
    """gd is gradient descent"""
    self.__optimizer_foreward = self.__gd_foreward
    self.__optimizer_backward = self.__gd_backward

  def __gd_foreward(self, X: np.ndarray) -> np.ndarray:
    return np.dot(X, self.__weights) + self.__bias

  def __gd_backward(self, gradient: np.ndarray) -> np.ndarray:
    self.__weights = self.__weights - self.__lr * np.dot(self.__X.T, gradient)
    self.__bias = self.__bias - self.__lr * np.sum(gradient, axis=0)
    return np.dot(gradient, self.__weights.T)

  def __nesterov_momentum_gd_init(self) -> None:
    """ Nesterov Gradient Descent """
    self.__vw: np.ndarray = np.zeros_like(self.__weights)
    self.__vb: np.ndarray = np.zeros_like(self.__bias)
    self.__optimizer_foreward = self.__nesterov_momentum_gd_foreward
    self.__optimizer_backward = self.__nesterov_momentum_gd_backward

  def __nesterov_momentum_gd_foreward(self, X: np.ndarray) -> np.ndarray:
    new_weights = self.__weights - self.__vw
    new_bias = self.__bias - self.__vb
    return np.dot(X, new_weights) + new_bias

  def __nesterov_momentum_gd_backward(self, gradient: np.ndarray) -> np.ndarray:
    self.__vw = self.__momentum_v * self.__vw + self.__lr * np.dot(self.__X.T, gradient)
    self.__vb = self.__momentum_v * self.__vb + self.__lr * np.sum(gradient, axis=0)
    self.__weights = self.__weights - self.__vw
    self.__bias = self.__bias - self.__vb
    return np.dot(gradient, self.__weights.T)

  def __rmsprop_init(self) -> None:
    self.__sw: np.ndarray = np.zeros_like(self.__weights)
    self.__sb: np.ndarray = np.zeros_like(self.__bias)
    self.__optimizer_foreward = self.__rmsprop_foreward
    self.__optimizer_backward = self.__rmsprop_backward

  def __rmsprop_foreward(self, X: np.ndarray) -> np.ndarray:
    return np.dot(X, self.__weights) + self.__bias

  def __rmsprop_backward(self, gradient: np.ndarray) -> np.ndarray:
    epsilon: float = 1e-9
    w_gradient = np.dot(self.__X.T, gradient)
    b_gradient = np.sum(gradient, axis=0)
    self.__sw = self.__momentum_s * self.__sw + (1 - self.__momentum_s) * np.square(w_gradient)
    self.__sb = self.__momentum_s * self.__sb + (1 - self.__momentum_s) * np.square(b_gradient)
    self.__weights = self.__weights - self.__lr * w_gradient / np.sqrt(self.__sw + epsilon)
    self.__bias    = self.__bias    - self.__lr * b_gradient / np.sqrt(self.__sb + epsilon)
    return np.dot(gradient, self.__weights.T)

  def __adam_init(self) -> None:
    self.__vw: np.ndarray = np.zeros_like(self.__weights)
    self.__vb: np.ndarray = np.zeros_like(self.__bias)
    self.__sw: np.ndarray = np.zeros_like(self.__weights)
    self.__sb: np.ndarray = np.zeros_like(self.__bias)
    self.__optimizer_foreward = self.__adam_foreward
    self.__optimizer_backward = self.__adam_backward

  def __adam_foreward(self, X: np.ndarray) -> np.ndarray:
    # new_weights = self.__weights - self.__vw
    # new_bias = self.__bias - self.__vb
    # return np.dot(X, new_weights) + new_bias
    return np.dot(X, self.__weights) + self.__bias

  def __adam_backward(self, gradient: np.ndarray) -> np.ndarray:
    mv, ms, epsilon = self.__momentum_v, self.__momentum_s, 1e-9
    w_gradient = np.dot(self.__X.T, gradient)
    b_gradient = np.sum(gradient, axis=0)
    self.__vw = mv * self.__vw + (1 - mv) * w_gradient
    self.__vb = mv * self.__vb + (1 - mv) * b_gradient
    self.__sw = ms * self.__sw + (1 - ms) * np.square(w_gradient)
    self.__sb = ms * self.__sb + (1 - ms) * np.square(b_gradient)
    vw_hat = self.__vw / (1 - mv)
    vb_hat = self.__vb / (1 - mv)
    sw_hat = self.__sw / (1 - ms)
    sb_hat = self.__sb / (1 - ms)
    self.__weights = self.__weights - self.__lr * vw_hat / np.sqrt(sw_hat + epsilon)
    self.__bias    = self.__bias    - self.__lr * vb_hat / np.sqrt(sb_hat + epsilon)
    return np.dot(gradient, self.__weights.T)

  def __call__(self, layer: Layer) -> None:
    """
    object call function for initialization: weights, biases, optimizer_init
    take layer for getting 'layer' output_shape as own input_shape
    """
    input_shape = layer.output_shape
    match self.__weights_initializer:
      case 'random':
        self.__weights: np.ndarray = np.random.randn(input_shape, self._output_shape)
        self.__bias   : np.ndarray = np.random.randn(self._output_shape)

      case 'uniform':
        max = 1 / np.sqrt(input_shape)
        self.__weights: np.ndarray = np.random.uniform(-max, max, size=(input_shape, self._output_shape))
        self.__bias   : np.ndarray = np.random.uniform(-max, max, size=self._output_shape)

      case 'xavier_uniform':
        max = np.sqrt(6 / (input_shape + self._output_shape))
        self.__weights: np.ndarray = np.random.uniform(-max, max, size=(input_shape, self._output_shape))
        self.__bias   : np.ndarray = np.random.uniform(-max, max, size=self._output_shape)

      case 'xavier_normal':
        std = np.sqrt(2 / (input_shape + self._output_shape))
        self.__weights: np.ndarray = np.random.normal(0, std, size=(input_shape, self._output_shape))
        self.__bias   : np.ndarray = np.random.normal(0, std, size=self._output_shape)

      case 'he_uniform':
        max = np.sqrt(6 / input_shape)
        self.__weights: np.ndarray = np.random.uniform(-max, max, size=(input_shape, self._output_shape))
        self.__bias   : np.ndarray = np.random.uniform(-max, max, size=self._output_shape)

      case 'he_normal':
        std = np.sqrt(2 / input_shape)
        self.__weights: np.ndarray = np.random.normal(0, std, size=(input_shape, self._output_shape))
        self.__bias   : np.ndarray = np.random.normal(0, std, size=self._output_shape)

      case _:
        raise NotImplementedError(f'DenseLayer __call__ not implemented [{self.__weights_initializer}] parameters initilizer')

  def backup(self) -> None:
    self.__weights_backup = self.__weights
    self.__bias_backup    = self.__bias

  def restore(self) -> None:
    self.__weights = self.__weights_backup
    self.__bias    = self.__bias_backup

  def init(self, optimizer: optimizer__type, lr: float, momentum_s: float, momentum_v: float) -> None:
    self.__lr         : float = lr
    self.__momentum_v : float = momentum_v
    self.__momentum_s : float = momentum_s
    match optimizer:
      case 'gd' | 'gradient_descent':
        self.__gd_init()
      case 'nag' | 'nmgd' | 'nesterov_momentum_gradient_descent':
        self.__nesterov_momentum_gd_init()
      case 'rmsprop':
        self.__rmsprop_init()
      case 'adam':
        self.__adam_init()
      case _:
        raise NotImplementedError(f'DenseLayer init not implemented [{optimizer}] optimizer')

  def save(self) -> DenseLayerData:
    return DenseLayerData(
      self.__weights, self.__bias, 
      self.__activation.activation,
      self._output_shape
      )

  def load(self, weights: np.ndarray, bias: np.ndarray) -> None:
    self.__weights = weights
    self.__bias = bias

  def predict(self, X: np.ndarray) -> np.ndarray:
    z =  np.dot(X, self.__weights) + self.__bias
    return self.__activation.foreward(z)

  def foreward(self, X: np.ndarray) -> np.ndarray:
    self.__X = X
    z = self.__optimizer_foreward(X)
    return self.__activation.foreward(z)

  def backward(self, gradient: np.ndarray, *, last: bool = False) -> np.ndarray:
    if not last:
      gradient = gradient * self.__activation.backward()
    return self.__optimizer_backward(gradient)
