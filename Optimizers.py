import optax
import jax
from Loss import MSE
from LossHandler import LossHandler
from functools import partial


class Optimizer:
    def __init__(self,lr: float, optimizer, ) -> None:
        self.lr = lr
        self.optimizer = optimizer

    def update(self, y_true, x, loss: callable = None, params: dict = {}):
        loss_value, grad = jax.value_and_grad(loss, allow_int = True)( params, y_true, x)
        updates, self.optimizer_state = self.optimizer.update(grad, self.optimizer_state, params)
        return optax.apply_updates(params, updates), loss_value
    
    def set_params(self, params):
        self.params = params
        self.optimizer_state = self.optimizer.init(self.params)

class Adam(Optimizer):
    def __init__(self, lr: float = 0.001, ) -> None:
        optimizer = optax.adam(learning_rate= lr, )
        super().__init__(lr, optimizer = optimizer, )


