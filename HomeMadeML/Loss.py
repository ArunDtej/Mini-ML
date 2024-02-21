import jax.numpy as np
from jax import random
import copy
import math
from collections.abc import Iterable
import optax


class Loss:
    def __init__(self,):
        """
        Initialize loss functions

        Args:
            loss (str)|(list): type of loss functions, can either be str or list of str or Loss objects made by extending the
            loss class with get_loss function defining the loss.
        """
        pass

    def __call__(self, y_true: float, y_pred: float) -> float:
        """
        Calculates the loss for the given labels and predictions. Implement your loss calculations in the loss function.

        Args:
            y_true (ndarray): True labels of the given data.
            y_pred (ndarray): predicted values for the given data.

        Returns:
            float: Numerical loss value.
        """
        raise NotImplementedError('Invalid Loss')


class MSE(Loss):
    def __init__(self,):
        """
        Initializes Mean squared error loss object.
        Call the loss function to get the loss for the given labels and predictions.
        """

    def __call__(self, y_true: float, y_pred: float) -> float:
        """
            Calculates MSE loss for the given labels and predictions.

            Args:
                y_true (float): True labels for the given data.
                y_pred (float): predicted values for the given data.

            Returns:
                float: Numerical MSE loss.
        """
        return np.mean(np.square(y_true - y_pred))


class MAE(Loss):
    def __init__(self,):
        """
        Initializes Mean Absolute error loss object.
        Call the loss function to get the loss for the given labels and predictions.
        """

    def __call__(self, y_true: float, y_pred: float) -> float:
        """
            Calculates MAE loss for the given labels and predictions.

            Args:
                y_true (float): True labels for the given data.
                y_pred (float): predicted values for the given data.

            Returns:
                float: Numerical MAE loss.
        """
        return np.mean(np.abs(y_true - y_pred))


class Kl_divergence(Loss):
    def __init__(self, epsilon=1e-7):
        """
            Initializes kl_divergence loss objet.
            Call the loss function to get the loss for the given labels and predictions.
            Args:
                epsilon (float): small numerical value for numerical stability.
        """
        self.epsilon = epsilon

    def __call__(self, y_true: float, y_pred: float) -> float:
        """
            Calculates Kl_divergence loss for the given labels and predictions.

            Args:
                y_true (float): True labels for the given data.
                y_pred (float): predicted values for the given data.

            Returns:
                float: Numerical Kl_divergence loss.
        """
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return np.mean(np.sum(y_true * np.log(y_true / y_pred), axis=-1))


class BinaryCrossentropy(Loss):
    def __init__(self, epsilon=1e-7):
        """
            Initializes BinaryCrossentropy loss object for classification problem.
            Call the loss function to get the loss for the given predictions and labels.
            Args:
                epsilon (float): small numerical value for numerical stability.
        """
        self.epsilon = 1e-7

    def __call__(self, y_true: int, y_pred: float, ) -> float:
        """
            Calculates BinaryCrossentropy loss for the given labels and predictions.

            Args:
                y_true (float): True labels for the given data.
                y_pred (float): predicted values for the given data.

            Returns:
                float: Numerical BinaryCrossentropy loss.
        """
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=-1))


class CategoricalCrossentropy(Loss):
    def __init__(self, epsilon=1e-7):
        """
            Initializes CategoricalCrossentropy loss object for classification problem.
            Call the loss function to get the loss for the given predictions and labels.
            Args:
                epsilon (float): small numerical value for numerical stability.
        """
        self.epsilon = 1e-7

    def __call__(self, y_true: int, y_pred: float, ) -> float:
        """
            Calculates CategoricalCrossentropy loss for the given labels and predictions.

            Args:
                y_true (float): True labels for the given data.
                y_pred (float): predicted values for the given data.

            Returns:
                float: Numerical CategoricalCrossentropy loss.
        """
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

class SparseCategoricalCrossentropy(Loss):
    """
            Initializes Sparse_Categorical_Crossentropy loss object for classification problem.
            Call the loss function to get the loss for the given labels and predictions. 
            Args:
                epsilon (float): small numerical value for numerical stability.
    """
    def __call__(self, y_true: int, y_pred: float,):
        """
            Calculates Sparse_Categorical_Crossentropy loss for the given labels and predictions.

            Args:
                y_true (float): True labels for the given data in onehot encoded format.
                y_pred (float): predicted values for the given data.

            Returns:
                float: Numerical loss value.
        """
        loss_value = optax.softmax_cross_entropy_with_integer_labels(logits = y_pred, labels=y_true.reshape(-1))
        return loss_value.mean()