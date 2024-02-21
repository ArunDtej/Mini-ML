from jax import numpy as jnp
import numpy as np
import jax


class Layer:
    """
      Layer Class containing various callable layers with trainable parameters.
    """

    def __init__(self, input_layer, Trainable: bool) -> None:
        """
        Initialize Layers class

        Args:
        - input_layer (Layer): previous layer.
        - Trainable (bool): set False to prevent training of Layers parameters.
        """
        self.input_layer = input_layer
        self.Trainable = Trainable
        if input_layer != None:
            self.input_layer.layercount = self.input_layer.layercount + 1
            self.layercount = self.input_layer.layercount
            self.params = self.input_layer.params
        else:
            self.layercount = 0
            self.params = {}
        self.layer_name = 'Layer {}'.format(self.layercount)

    def set_trainables(self) -> None:
        """
        Adds trainable parameters from layer to model parameters, if Trainable is set to 'True'
        """
        if self.Trainable:
            self.params.update(self.sub_params)

    def display(self) -> None:
        print('Layer Class')

    def __call__(self, x, params) -> None:
        raise NotImplementedError(self.error_message)


class InputLayer(Layer):
    """
          Initialize Input layer, its must for every model to have its first layer as InputLayer.

          Args:
          - input_layer (Layer): Previous layerif it takes its inputs from any layer, if not then left with None.
          """

    def __init__(self, input_layer: None = None):
        self.layercount = 0
        super().__init__(input_layer, Trainable=False)

    def __call__(self, x, params):
        return x

    def display(self):
        print('Input Layer',)


class Relu(Layer):
    def __init__(self, input_layer, Trainable: bool = False) -> None:
        """
          Initialize Relu activation.

          Args:
          - input_layer (Layer): previous layer.
          - Trainable (bool): set False to prevent training of Layers parameters.
          """
        self.activation = 'ReLU'
        super().__init__(input_layer, Trainable=False)
        self.sub_params = {}
        self.set_trainables()

    def __call__(self, x, params: dict):
        """
              Applies Relu activation to the inputs.

              Args:
              - x (Array): Inputs.

              Returns:
              - Activated outputs.
          """
        self.outputs = jnp.maximum(0, x)
        return self.outputs

    def display(self):
        print('Activation : ', self.activation)


class Sigmoid(Layer):
    def __init__(self, input_layer=None, Trainable: bool = False) -> None:
        """
          Initialize Sigmoid activation.

          Args:
          - input_layer (Layer): previous layer.
          - Trainable (bool): set False to prevent training of Layers parameters.
          """
        self.activation = 'Sigmoid'
        super().__init__(input_layer, Trainable=False)
        self.sub_params = {}
        self.set_trainables()

    def __call__(self, x, params: dict):
        """
              Applies Sigmoid activation to the inputs.

              Args:
              - x (Array): Inputs.

              Returns:
              - Activated outputs.
          """
        self.outputs = 1 / (1 + jnp.exp(-x))
        return self.outputs

    def display(self):
        print('Activation : ', self.activation)


class Tanh(Layer):
    def __init__(self, input_layer, Trainable: bool = False) -> None:
        """
          Initialize Leaky relu activation.

          Args:
          - input_layer (Layer): previous layer.
          - Trainable (bool): set False to prevent training of Layers parameters.
          """
        self.activation = 'Tanh'
        super().__init__(input_layer, Trainable=False)
        self.sub_params = {}
        self.set_trainables()

    def __call__(self, x, params: dict):
        """
              Applies Tanh activation to the inputs.

              Args:
              - x (Array): Inputs.

              Returns:
              - Activated outputs.
          """
        self.outputs = jnp.tanh(self.x)
        return self.outputs

    def display(self):
        print('Activation : ', self.activation)


class Softmax(Layer):
    def __init__(self, input_layer, Trainable: bool = False) -> None:
        """
          Initialize Leaky relu activation.

          Args:
          - input_layer (Layer): previous layer.
          - Trainable (bool): set False to prevent training of Layers parameters.
          """
        self.activation = 'Softmax'
        self.input_layer = input_layer
        super().__init__(input_layer, Trainable=Trainable)
        self.sub_params = {}
        self.set_trainables()

    def __call__(self, x, params: dict):
        """
              Applies Softmax activation to the inputs.

              Args:
              - x (Array): Inputs.

              Returns:
              - Activated outputs.
          """
        exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
        return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

    def display(self):
        print('Activation : ', self.activation)


class LeakyRelu(Layer):
    def __init__(self, input_layer=None, alpha: float = 0.01, Trainable: bool = True) -> None:
        """
          Initialize Leaky relu activation.

          Args:
          - input_layer (Layer): previous layer.
          - Trainable (bool): set False to prevent training of Layers parameters.
          """
        self.alpha = alpha
        super().__init__(input_layer, Trainable=True, )
        self.trainable_parameters = [self.layer_name]
        self.sub_params = {
            self.trainable_parameters[0]: jnp.array([self.alpha])
        }
        self.set_trainables()
        self.activation = 'LeakyReLU'

    def __call__(self, x, params: dict = {}):
        """
              Applies Leaky Relu activation to the inputs.

              Args:
              - x (Array): Inputs.

              Returns:
              - Activated outputs.
          """
        alpha = self.sub_params[self.trainable_parameters[0]]

        if self.Trainable and (set(self.trainable_parameters).issubset(params.keys())):
            alpha = params[self.trainable_parameters[0]]
        self.outputs = jnp.maximum(alpha * x, x)
        return self.outputs

    def display(self):
        print('Activation : ', self.activation)


class Dense(Layer):
    def __init__(self, input_layer, input_shape: int, output_shape: int, Trainable: bool = True) -> None:
        """
        Initialize a Dense layer with weights and bias for the given input and output shape

        Args:
          input_shape (list): shape of the input.
          output_shape (list): shape of the outputs.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        super().__init__(input_layer, Trainable=Trainable)
        self.trainable_parameters = ['weights {}'.format(
            self.layer_name), 'bias {}'.format(self.layer_name)]
        self.sub_params = {
            self.trainable_parameters[0]: jax.random.normal(jax.random.PRNGKey(1), (input_shape, output_shape)),
            self.trainable_parameters[1]: jax.random.normal(jax.random.PRNGKey(1), (1, output_shape)),
        }
        self.set_trainables()

    def __call__(self, x, params: dict = {}):
        """
              Applies Dense Layer forward propaagation.

              Args:
              - x (Array): Inputs.
              - params (dict) : Trainable parameters dict. To be left alone as its maintained by teh model itself.

              Returns:
              - ndarray.
          """

        weights = self.sub_params[self.trainable_parameters[0]]
        bias = self.sub_params[self.trainable_parameters[1]]

        if self.Trainable and (set(self.trainable_parameters).issubset(params.keys())):
            weights = params[self.trainable_parameters[0]]
            bias = params[self.trainable_parameters[1]]
        self.outputs = jnp.dot(x, weights) + bias
        return self.outputs

    def display(self):
        print('Dense Layer')
        print('learnable weights :', (self.weights).shape)
        print('learnable bias :', (self.bias).shape)


class Dropout(Layer):
    def __init__(self, fraction, input_layer, Trainable=False):
        """
        Initialize Dropout layer with the given fraction.

        Args:
        - fraction (list): fraction of nodes to be dropped out in a given layer.
        """
        self.input_layer = input_layer
        super().__init__(input_layer, Trainable=Trainable)
        self.sub_params = {}
        self.set_trainables()
        self.fraction = fraction

    def __call__(self, x, params: dict):
        """
        Applies to the given input.

        Args:
              - x (Array): Inputs.
              - params (dict) : Trainable parameters dict. To be left alone as its maintained by teh model itself.

              Returns:
              - ndarray.
        """
        num_nodes = x.shape[-1]
        num_samples = int(num_nodes * self.fraction)
        indices = np.random.choice(num_nodes, size=num_samples, replace=False)
        x[..., indices] = 0
        return x

    def display(self):
        print('Dropout', self.fraction)
