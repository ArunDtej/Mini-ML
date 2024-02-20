from collections.abc import Iterable
import Loss
import jax.numpy as jnp


class LossHandler:
    """Loss handler object to handle loss calculation and weighted loss calculation."""

    def __init__(self, model, losses: list, weights: list = []) -> None:
        """
        initialize the LossHandler parameters
        Args:
        - model : Callable model object.
        - losses (list): iterable loss objects, losses should be an instance of the Loss class.
        - weights (list): weights corresponding to the given loss, if none then weights are set equally.
        """
        self.model = model
        self.losses = losses
        self.weights = weights

    def get_predictions(self, x):
        """
        Gets predictions from the model to calculate the loss.
        Args:
        - x : inputs.

        Returns:
        - predictions
        """
        return self.model(x)

    def __call__(self, params, y_true, x, ):
        """
        Calculates the weighted loss for the given loss objects.

        Args:
        - y_true (np.ndarray): True outputs.
        - x (np.nparray): inputs.

        Returns:
        - int: loss for the labels and predictions.
        """
        y_hat = self.model( x, params)
        if len(self.weights) == 0:
            self.weights = jnp.ones((len(self.losses)))

        loss = 0
        for sub_loss, weight in zip(self.losses, self.weights):
            if isinstance(sub_loss, Loss.Loss):
                loss += sub_loss( y_true, y_hat)*weight
            else:
                message = "Unrecognized loss object {} will be ignored.".format(
                    type(sub_loss))
                print(message)

        return loss
