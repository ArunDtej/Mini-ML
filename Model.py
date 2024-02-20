
from Layer import Layer
from Trainer import Trainer
from LossHandler import LossHandler
from Optimizers import Adam

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
        previous_layer = None
        for layer in self.structure:
            if isinstance(layer, Layer):
                self.Model.append(layer)
            else:
                raise ValueError("Unknown layer object")
        self.params = self.Model[-1].params

    def compile(self, optimizer = Adam(lr=0.001), losses: list =['mse'], weights:list = [], metrics=None):
        self.losshandler = LossHandler(model = self, losses=losses, weights = weights)
        self.optimizer = optimizer
        self.optimizer.set_params(self.params)
        self.trainer = Trainer(self.optimizer, self.losshandler, metrics, self.params)

    def train(self, train_data, epochs, lr=0.001, batch_size=32, valid_data=None, train_steps=None,
              test_steps=None, steps=None, ):
        for epoch in range(epochs):
            print('Epoch : {}'.format(epoch))
            self.params = self.trainer(self, train_data, epochs, lr, batch_size, )

    def __call__(self, x, params = None):
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
