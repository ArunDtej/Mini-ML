Mini Deep Learning framework using jax as backend for weight optimization, supports gradient tracking on custom parameters to empower extendability.
Currently supports Dense Layers, if you want to implement any new nonexisting layers then feel free to make a pull request,
This is soley a learning project that I made out of ethusiaism, so don't expect much. 
It works fine according to my experimentations, so good luck trying it out.

# **LAYERS**

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

LOSSHANDLER

LossHandler:
Description: A class to handle loss calculation and weighted loss calculation for a given model.
Attributes:
model: Callable model object for which the loss is being calculated.
losses (list): Iterable containing loss objects. Each loss should be an instance of the Loss class.
weights (list): Weights corresponding to the given loss objects. If not provided, weights are set equally.
Methods:
__init__(model, losses, weights): Initializes the LossHandler object with the model, losses, and optional weights.
get_predictions(x): Returns predictions from the model for a given input x.
__call__(params, y_true, x): Calculates the weighted loss for the given loss objects and inputs.
params: Parameters of the model.
y_true: True outputs.
x: Inputs.
Note: If an unrecognized loss object is encountered, a message is printed, and that loss is ignored.

MODEL

HomeMadeModel:
Description: A custom-built neural network model class with configurable layers and training methods.
Attributes:
input_shape: Shape of the input data that is fed into the network.
structure (list): List of dictionaries, each specifying the type of layer, activation, and dropout.
Model (list): List of layer objects comprising the model.
params: Parameters of the model.
losshandler: LossHandler object for handling loss calculation.
optimizer: Optimizer object for updating model parameters.
trainer: Trainer object for training the model.
Methods:
__init__(input_shape, structure): Initializes the model with the given input shape and structure.
init_parameters(): Initializes the network parameters based on the specified structure.
compile(optimizer, losses, weights, metrics): Compiles the model with the specified optimizer, losses, weights, and metrics.
train(train_data, epochs, lr, batch_size, valid_data, train_steps, test_steps, steps): Trains the model on the given training data for the specified number of epochs.
__call__(x, params): Executes the model on input x with the specified parameters.
display(): Displays information about the model's input shape and each layer's configuration.

OPTIMIZER

Optimizer:

Description: A generic optimizer class for updating model parameters using various optimization algorithms.
Attributes:
lr: Learning rate for the optimizer.
optimizer: Optimizer object from the Optax library.
Methods:
__init__(lr, optimizer): Initializes the optimizer with the given learning rate and optimizer object.
update(y_true, x, loss, params): Updates the model parameters based on the loss function and gradients.
set_params(params): Sets the model parameters and initializes the optimizer state.
Adam:

Description: An Adam optimizer subclass inheriting from the Optimizer class, specifically configured for Adam optimization.
Attributes:
Inherits lr and optimizer from the Optimizer class.
Methods:
__init__(lr): Initializes the Adam optimizer with the given learning rate.
Inherits update and set_params methods from the Optimizer class.

TRAINER

Trainer:
Description: A class for training a model using a specified optimizer and loss handler.
Attributes:
optimizer: An optimizer object used to update the model parameters.
losshandler: A loss handler object used to calculate the loss for the model.
metrics: Metrics to evaluate the model's performance during training.
params: The model's parameters.
epoch_loss: Cumulative loss for the current epoch.
epoch: Current epoch number.
epoch_progress_bar: Progress bar for tracking the training progress within an epoch.
num_batches: Number of batches processed in the current epoch.
Methods:
__init__(optimizer, losshandler, metrics, params): Initializes the Trainer with the given optimizer, loss handler, metrics, and model parameters.
__call__(Model, train_data, epochs, lr, batch_size, valid_data=None, train_steps=None, test_steps=None, steps=None): Trains the model for the specified number of epochs using the given data.
Train(x, y): Performs training on a single batch of data, updating the model's parameters and tracking the loss.
