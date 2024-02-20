from Loss import MSE, MAE
from LossHandler import LossHandler
import Layer
import numpy as np
import Optimizers

# l1 = Layer.Dense(None, 3,3)
l1 = Layer.LeakyRelu(input_layer=None)

x = np.random.randn(10,3)
# print(x==l1(x))

def f(x):
    return 5*x-10

y = f(x)
# y_pred = l1(x)

params = l1.params

print('Params : {}'.format(params))
losshandler = LossHandler(l1, losses = [MSE(), MAE()] )

print(callable(losshandler))

optimizer = Optimizers.Adam(loss = losshandler, params = params)
params = optimizer.update(y, x)

print(params)