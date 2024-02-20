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

LOSS

Loss:
Description: Base class for defining various loss functions used in training neural networks.
Methods:
__init__(): Initializes the loss function object.
__call__(y_true, y_pred): Calculates the loss between true labels y_true and predicted values y_pred. This method should be implemented in subclasses.

MSE (Mean Squared Error):
Description: Subclass of Loss representing the Mean Squared Error loss function.
Methods:
__init__(): Initializes the MSE loss function object.
__call__(y_true, y_pred): Calculates the Mean Squared Error loss between y_true and y_pred.

MAE (Mean Absolute Error):
Description: Subclass of Loss representing the Mean Absolute Error loss function.
Methods:
__init__(): Initializes the MAE loss function object.
__call__(y_true, y_pred): Calculates the Mean Absolute Error loss between y_true and y_pred.

Kl_divergence:
Description: Subclass of Loss representing the Kullback-Leibler divergence loss function.
Methods:
__init__(epsilon): Initializes the KL divergence loss function object with a small numerical value epsilon.
__call__(y_true, y_pred): Calculates the KL divergence loss between y_true and y_pred.

BinaryCrossentropy:
Description: Subclass of Loss representing the Binary Crossentropy loss function for binary classification.
Methods:
__init__(epsilon): Initializes the Binary Crossentropy loss function object with a small numerical value epsilon.
__call__(y_true, y_pred): Calculates the Binary Crossentropy loss between y_true and y_pred.

CategoricalCrossentropy:
Description: Subclass of Loss representing the Categorical Crossentropy loss function for multi-class classification.
Methods:
__init__(epsilon): Initializes the Categorical Crossentropy loss function object with a small numerical value epsilon.
__call__(y_true, y_pred): Calculates the Categorical Crossentropy loss between y_true and y_pred.

Sparse_Categorical_Crossentropy:
Description: Subclass of Loss representing the Sparse Categorical Crossentropy loss function for multi-class classification with integer labels.
Methods:
__call__(y_true, y_pred): Calculates the Sparse Categorical Crossentropy loss between y_true and y_pred.
