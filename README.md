Mini Deep Learning framework using jax as backend for weight optimization, supports gradient tracking on custom parameters to empower extendability.
Currently supports Dense Layers, if you want to implement any new nonexisting layers then feel free to make a pull request,
This is soley a learning project that I made out of ethusiaism, so don't expect much. 
It works fine according to my experimentations, so good luck trying it out.

LAYERS

Layer:
Description: Base class for creating neural network layers with trainable parameters.
Methods:
__init__(input_layer, Trainable): Initializes the layer with a previous layer and sets whether it is trainable.
set_trainables(): Adds trainable parameters to the model parameters if the layer is trainable.
display(): Displays the type of layer.
__call__(x, params): Raises a NotImplementedError to be implemented in subclasses.

InputLayer:
Description: Subclass of Layer representing an input layer that returns its input unchanged.
Methods:
__init__(input_layer): Initializes the input layer.
__call__(x, params): Returns the input unchanged.
display(): Displays the type of layer.

Relu:
Description: Subclass of Layer representing a ReLU activation function.
Methods:
__init__(input_layer, Trainable): Initializes the ReLU activation function.
__call__(x, params): Applies the ReLU activation to the inputs.
display(): Displays the type of activation function.

Sigmoid:
Description: Subclass of Layer representing a Sigmoid activation function.
Methods:
__init__(input_layer, Trainable): Initializes the Sigmoid activation function.
__call__(x, params): Applies the Sigmoid activation to the inputs.
display(): Displays the type of activation function.

Tanh:
Description: Subclass of Layer representing a Tanh activation function.
Methods:
__init__(input_layer, Trainable): Initializes the Tanh activation function.
__call__(x, params): Applies the Tanh activation to the inputs.
display(): Displays the type of activation function.

Softmax:
Description: Subclass of Layer representing a Softmax activation function.
Methods:
__init__(input_layer, Trainable): Initializes the Softmax activation function.
__call__(x, params): Applies the Softmax activation to the inputs.
display(): Displays the type of activation function.

LeakyRelu:
Description: Subclass of Layer representing a Leaky ReLU activation function.
Methods:
__init__(input_layer, alpha, Trainable): Initializes the Leaky ReLU activation function with a given alpha.
__call__(x, params): Applies the Leaky ReLU activation to the inputs.
display(): Displays the type of activation function.

Dense:
Description: Subclass of Layer representing a Dense (fully connected) layer.
Methods:
__init__(input_layer, input_shape, output_shape, Trainable): Initializes the Dense layer with weights and biases.
__call__(x, params): Applies the Dense layer forward pass to the inputs.
display(): Displays information about the Dense layer.

Dropout:
Description: Subclass of Layer representing a Dropout layer.
Methods:
__init__(fraction, input_layer, Trainable): Initializes the Dropout layer with a dropout fraction.
__call__(x, params): Applies dropout to the inputs.
display(): Displays information about the Dropout layer.
