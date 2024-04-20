from mlp import MLP 
from tensor import Tensor 

class Optimizer:

    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()

    def step(self):
        for param in self.parameters:
            param.data -= param.grad * self.lr
