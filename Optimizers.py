import optax
import jax
from Loss import MSE
from LossHandler import LossHandler
from functools import partial


class Optimizer:
    def __init__(self,lr: float, optimizer, ) -> None:
        """
        Initialize Optimizer with the given lr and optimizer type.

        Args:
          Structure (list): Structure is a list of dicts each with Type of layer, Activation and Dropout keys and values.
          input_shape (str): shape of the input data that is fed into the network.
        """
        self.lr = lr
        self.optimizer = optimizer

    def update(self, y_true, x, loss: callable = None, params: dict = {}):
        """
        Updates the parameters of the model, given as the params dict.

            Args:
            - y_true (ndarray): true labels
            - x (ndarray) : inputs.
            - loss(Callable) : loss handler object.
            - params (dict) : Trainable parameters, to be left None.

            Returns:
            - Updated parameters
        """
        loss_value, grad = jax.value_and_grad(loss, allow_int = True)( params, y_true, x)
        updates, self.optimizer_state = self.optimizer.update(grad, self.optimizer_state, params)
        return optax.apply_updates(params, updates), loss_value
    
    def set_params(self, params):
        self.params = params
        self.optimizer_state = self.optimizer.init(self.params)

class Adam(Optimizer):
    def __init__(self, lr: float = 0.001, ) -> None:
        """
        Initialize the Adam optimizer.with given learning rate (lr).
        """
        optimizer = optax.adam(learning_rate= lr, )
        super().__init__(lr, optimizer = optimizer, )


