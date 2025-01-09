import math
from typing import NoReturn, Iterator, Any, Literal, TypeAlias

import numpy as np

from .layers import Layer, Input, TrainLayer, Dropout, Dense
from .losses import Loss
from .utils import (
        ALL_ACTIVATIONS,
        ALL_LAYER_TYPES,
        ALL_LOSSES,
        ALL_OPTIMIZERS,
        DenseLayerData,
        DropoutLayerData,
        History,
        LayerTypeName,
        Metrics,
        OptimizerName,
        LossName
    )


class Sequential:
    @staticmethod
    def __split_validation_data(
        x: np.ndarray, y: np.ndarray, val_split: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert 0 < val_split < 1, Exception(
            f"Sequential: {val_split} validation split not valid, must be between 0 and 1"
        )
        data_size: int = x.shape[0]
        train_size: int = int(data_size * (1 - val_split))
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return x[:train_size], y[:train_size], x[train_size:], y[train_size:]

    @staticmethod
    def __batches(
        x: np.ndarray, y: np.ndarray, batch_size: int
    ) -> Iterator[tuple[int, np.ndarray, np.ndarray, int]]:
        data_size = x.shape[0]
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        id = start = 0
        while start < data_size:
            end = min(data_size, start + batch_size)
            yield id, x[start:end], y[start:end], end - start
            id += 1
            start = end

    def __init__(self, input_layer: Input | None = None, *layers: TrainLayer) -> None:
        self.__layers: list[TrainLayer]
        self.__loss: Loss
        self.__loss_name: LossName
        self.__optimizer: OptimizerName
        self.__compiled: bool = False
        self.__loaded: bool = False
        if input_layer is not None:
            self.__create_model_network(input_layer, *layers)

    def __create_model_network(self, input_layer: Input, *layers: TrainLayer) -> None | NoReturn:
        if not layers:
            raise Exception("Sequential: Expected non empty layers")
        prev_layer: Layer = input_layer
        for layer in layers:
            layer(prev_layer)
            prev_layer = layer
        self.__layers = list(layers)

    def __init_layers(self, optimizer: OptimizerName, lr: float, beta1: float, beta2: float) -> None:
        for layer in self.__layers:
            layer.init(optimizer, lr, beta1, beta2)

    def __early_stopping(
        self, loss: float, best_loss: float, patience: int, remaining_patience: int
    ) -> tuple[Literal["continue", "stopping", "breaking"], int]:
        if loss < best_loss:
            for layer in self.__layers:
                layer.backup()
            return "continue", patience
        if remaining_patience <= 0:
            for layer in self.__layers:
                layer.restore()
            return "breaking", patience
        return "stopping", remaining_patience - 1

    def __forward(self, x: np.ndarray, *, train: bool) -> np.ndarray:
        for layer in self.__layers:
            x = layer.forward(x, train=train)
        return x

    def __backward(self, gradient: np.ndarray) -> None:
        last: bool = True
        for layer in reversed(self.__layers):
            gradient = layer.backward(gradient, last=last)
            last = False

    def save(self, model_name: str = "./model", *, verbose: bool = True) -> None:
        loss_idx = ALL_LOSSES.index(self.__loss_name)
        optimizer_idx = ALL_OPTIMIZERS.index(self.__optimizer)
        model: dict[str, Any] = {
            "loss": loss_idx,
            'optimizer': optimizer_idx,
            "size": len(self.__layers),
        }
        for i, layer in enumerate(self.__layers):
            data = layer.save()
            type_idx = ALL_LAYER_TYPES.index(data.type)
            if isinstance(data, DropoutLayerData):
                model[f"output_dim-{i}"] = data.output_dim
                model[f"ratio-{i}"] = data.ratio
                model[f"type-{i}"] = type_idx
            elif isinstance(data, DenseLayerData):
                activation_idx = ALL_ACTIVATIONS.index(data.activation)
                model[f"w-{i}"] = data.weights
                model[f"b-{i}"] = data.bias
                model[f"output_dim-{i}"] = data.output_dim
                model[f"activation-{i}"] = activation_idx
                model[f"type-{i}"] = type_idx

        np.savez_compressed(model_name, **model)
        if verbose:
            print(f"> saving model '{model_name}.npz to disk...")

    def load(self, filename: str) -> None:
        if not filename.endswith(".npz"): filename = f"{filename}.npz"
        model = np.load(filename, allow_pickle=False)
        loss_idx = model["loss"]
        optimizer_idx = model["optimizer"]
        model_size = model["size"]
        loss_name = ALL_LOSSES[loss_idx]
        optimizer = ALL_OPTIMIZERS[optimizer_idx]
        self.__layers = []
        for i in range(model_size):
            type_idx = model[f"type-{i}"]
            type: LayerTypeName = ALL_LAYER_TYPES[type_idx]
            if type == "dropout":
                ratio = model[f"ratio-{i}"]
                output_dim = model[f"output_dim-{i}"]
                layer = Dropout(ratio=ratio, output_dim=output_dim)
                self.__layers.append(layer)
            elif type == "dense":
                weights = model[f"w-{i}"]
                bias = model[f"b-{i}"]
                output_dim = model[f"output_dim-{i}"]
                activation_idx = model[f"activation-{i}"]
                activation = ALL_ACTIVATIONS[activation_idx]
                layer = Dense(output_dim=output_dim, activation=activation)
                layer.load(weights=weights, bias=bias)
                self.__layers.append(layer)
        self.__loss_name = loss_name
        self.__optimizer = optimizer
        self.__loss = Loss(loss_name)
        self.__loaded = True

    def test(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        yHat = self.__forward(x, train=False)
        loss = self.__loss.loss(y, yHat)
        m1, m2, m3, m4 = self.__loss.metrics(y, yHat)
        match self.__loss_name:
            case "mse":
                return {
                    'loss': loss,
                    'mse': m1,
                    'rmse': m2,
                    'mae': m3
                }
            case "binarycrossentropy":
                return {
                    "loss": loss,
                    "accuracy": m1,
                    "precision": m2,
                    "recall": m3,
                    "f1_score": m4,
                }
            case "crossentropy":
                return {
                    "loss": loss,
                    "accuracy": m1,
                    "precision": m2,
                }

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        y_pred = self.__forward(x, train=False)
        match self.__loss_name:
            case "mse":
                return y_pred.reshape(-1)
            case "binarycrossentropy":
                return y_pred.reshape(-1)
            case "crossentropy":
                return y_pred

    def predict(self, x: np.ndarray) -> np.ndarray:
        y_pred = self.__forward(x, train=False)
        match self.__loss_name:
            case "mse":
                return y_pred.reshape(-1)
            case "binarycrossentropy":
                return (y_pred > 0.5).astype(int).reshape(-1)
            case "crossentropy":
                return y_pred.argmax(axis=1).reshape(-1)

    def compile(
        self, *,
        loss: LossName,
        optimizer: OptimizerName = "adam",
        lr=0.001,
        beta1=0.9,
        beta2=0.99
    ) -> None:
        self.__loss = Loss(loss=loss)
        self.__loss_name = loss
        self.__optimizer = optimizer
        self.__init_layers(self.__optimizer, lr, beta1, beta2)
        self.__compiled = True

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        batch_size : int = 100,
        epochs : int = 13,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        val_split : float = 0.0,
        enable_metrics: bool = False,
        early_stop : bool = False,
        early_stop_patience : int = 1,
        verbose : int = 1,
    ) -> History:

        history: History = History()
        val_data_exist: bool = False
        batch_count = math.ceil(x.shape[0] / batch_size)
        best_val_loss: float = float("inf")
        remaining_patience: int = early_stop_patience
        epochs_str = loss_acc_str = early_stop_str = ""

        if self.__loaded and not self.__compiled:
            raise Exception(
                "Sequential: "
                "the model is loaded from npz file.\n"
                "recomended to use .compile before fit (train again) "
                "to adapt learning rate, beta1, beta2, ....\n"
                f"use '{self.__loss_name}' as loss\n"
                f"optimizer of loaded model is '{self.__optimizer}', "
                "without .compile, default optimizer is gradient descent (gd)"
            )

        if not self.__compiled:
            raise Exception('Sequential: model not compiled yet, use .compile before fit')

        assert 0 <= val_split < 1, Exception(
            "Sequential: validation split (val_split)"
            " shall between 0 and 1, (0 <= val_split < 1)"
        )

        if x_val is None or y_val is None:
            if val_split != 0:
                x, y, x_val, y_val = self.__split_validation_data(x, y, val_split)
                assert y.ndim == y_val.ndim == 2, Exception("Sequential: y and y_val shall be in 2 dimentions")
                val_data_exist = True
            elif early_stop:
                raise Exception("Sequential: cannot enable early stop with no validation data")
        else:
            assert y.ndim == y_val.ndim == 2, Exception("Sequential: y and y_val shall be in 2 dimentions")
            val_data_exist = True

        def __get_loss_matric_history_train_only(y: np.ndarray, yHat: np.ndarray) -> tuple[float, float]:
            train_loss = self.__loss.loss(y, yHat)
            history.add_loss(train_loss)
            if enable_metrics:
                metrics = self.__loss.metrics(y, yHat)
                history.add_metrics(metrics)
                return train_loss, metrics[0]
            return train_loss, 0.0

        def __get_loss_matric_history_train_and_validation(y: np.ndarray, yHat: np.ndarray) -> Metrics:
            yHat_val = self.__forward(x_val, train=False) # type: ignore
            train_loss = self.__loss.loss(y, yHat)
            val_loss = self.__loss.loss(y_val, yHat_val) # type: ignore
            history.add_loss(train_loss)
            history.add_val_loss(val_loss)
            if enable_metrics:
                metrics = self.__loss.metrics(y, yHat)
                val_metrics = self.__loss.metrics(y_val, yHat_val) # type: ignore
                history.add_metrics(metrics)
                history.add_val_metrics(val_metrics)
                return train_loss, val_loss, metrics[0], val_metrics[0]
            return train_loss, val_loss, 0.0, 0.0

        def __handle_loss_accuracy_history_verbose_str(y: np.ndarray, yHat: np.ndarray) -> tuple[str, float]:
            if val_data_exist:
                los, v_los, acc, v_acc = __get_loss_matric_history_train_and_validation(y, yHat)
                verbose_str = f"loss: {los:.9f}  val_loss: {v_los:.9f}"
                if enable_metrics:
                    verbose_str += f"  accuracy: {acc:.9f}  val_accuracy: {v_acc:.9f}"
            else:
                los, acc = __get_loss_matric_history_train_only(y, yHat)
                verbose_str  = f"loss: {los:.9f}"
                if enable_metrics:
                    verbose_str += f"  accuracy: {acc:.9f}"
                v_los = best_val_loss
            return verbose_str, v_los

        for epoch in range(epochs):
            for id, xb, yb, size in self.__batches(x, y, batch_size):
                yHat = self.__forward(xb, train=True)
                gradient = self.__loss.derivative(yb, yHat, size)
                self.__backward(gradient)

                loss_acc_str, val_loss = __handle_loss_accuracy_history_verbose_str(yb, yHat)
                if early_stop:
                    early_stop_status, remaining_patience = self.__early_stopping(
                        val_loss, best_val_loss, early_stop_patience, remaining_patience
                    )
                    early_stop_str = f" early_stop: {early_stop_status}"
                    best_val_loss = val_loss
                    if early_stop_status == "breaking":
                        pass

                epochs_str = f"{epoch: 4}/{epochs}"
                batch_str = f"({id: 4}/{batch_count})"

                if verbose == 2:
                    print(f"{epochs_str} {batch_str}  {loss_acc_str}{early_stop_str}")

            if verbose == 1:
                print(f"{epochs_str} ({batch_count})  {loss_acc_str}{early_stop_str}")

        return history

