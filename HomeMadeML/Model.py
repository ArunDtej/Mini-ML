
from HomeMadeML.Layer import Layer
from HomeMadeML.Trainer import Trainer
from HomeMadeML.LossHandler import LossHandler
from HomeMadeML.Optimizers import Adam

class HomeMadeModel:
    def __init__(self, input_shape = None, structure: list = []):
        """
        Initialize Model with the given Structure.

        Args:
          Structure (list): Structure is a list of dicts each with Type of layer, Activation and Dropout keys and values.
          input_shape (str): shape of the input data that is fed into the network.
        """
        assert len(structure) > 0, "Model should contain atleast one layer"

        self.input_shape = input_shape
        self.structure = structure
        self.Model = []
        self.init_parameters()

    def init_parameters(self):
        """Init the network parameters"""
        for layer in self.structure:
            if isinstance(layer, Layer):
                self.Model.append(layer)
            else:
                raise ValueError("Unknown layer object")
        # Getting trainable parameters that are to be tracked which are at the last layer of the model.
        self.params = self.Model[-1].params

    def compile(self, optimizer = Adam(lr=0.001), losses: list =['mse'], weights:list = [], metrics=None):
        """Building loss handler to handle multiple loss and getting the loss from predictions and true labels"""
        self.losshandler = LossHandler(model = self, losses=losses, weights = weights)
        self.optimizer = optimizer
        self.optimizer.set_params(self.params)
        # Trainer handles the batchwise data processing, predictions and optimizer calls while optimizer deals with updating the parameters
        self.trainer = Trainer(self.optimizer, self.losshandler, metrics, self.params)

    def train(self, train_data, epochs, lr=0.001, batch_size=32, valid_data=None, train_steps=None,
              test_steps=None, steps=None, ):
        for epoch in range(epochs):
            print('Epoch : {}'.format(epoch))
            self.params = self.trainer(self, train_data, epochs, lr, batch_size, )

    def __call__(self, x, params = None):
        """
            Args:
            - x (ndarray) : inputs.
            - params (dict) : Trainable parameters, to be left None.

            Returns:
            - y (ndarray): Model outputs.
        """
        if params==None:
            params = self.params
        for i in self.Model:
            x = i(x, params)
        return x

    def display(self):
        print('input shape :', self.input_shape, '\n')
        for i in self.Model:
            i.display()
            print()
