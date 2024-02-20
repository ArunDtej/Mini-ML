from tqdm import tqdm
from LossHandler import LossHandler


class Trainer:
    def __init__(self, optimizer, losshadler, metrics, params):
        self.optimizer = optimizer
        self.metrics = metrics
        self.epoch_loss = 0
        self.epoch = 0
        self.epoch_progress_bar = None
        self.params = params
        self.losshandler = losshadler

    def __call__(self, Model, train_data, epochs, lr, batch_size, valid_data=None, train_steps=None, test_steps=None, steps=None):

        self.num_batches = 0
        self.epoch_loss = 0

        training_len = len(train_data['y'])

        if type(train_data) == dict:
            self.epoch_progress_bar = tqdm(
                total=training_len//batch_size, position=0)
            for i in range(0, training_len, batch_size):
                x = train_data['x'][i:i+batch_size]
                y = train_data['y'][i:i+batch_size]
                self.Train(x, y)

        else:
            self.epoch_progress_bar = tqdm(
                total=train_steps, desc=f'Epoch {self.epoch}', position=0)
            for _ in range(train_steps):
                x, y = next(train_data)
                self.Train(x, y)

        self.epoch_progress_bar.close()
        return self.params

    def Train(self, x, y):
        # Training Model
        self.params, loss_value = self.optimizer.update(
            y, x, self.losshandler, self.params)
        self.epoch_loss += loss_value

        self.num_batches += 1
        self.epoch_progress_bar.set_postfix(
            {'Loss': self.epoch_loss/self.num_batches})
        self.epoch_progress_bar.update(1)
